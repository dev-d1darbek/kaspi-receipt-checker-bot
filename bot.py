"""Kaspi чек тексеру Telegram боты — бір файл.

Логика:
1. Бірінші чек → сатушы компаниясы (атауы + ЖСН/БСН) есте сақталады.
2. Кейінгі чектер тексеріледі:
   • Kaspi-де түпнұсқа ма
   • Сол компанияға тиесілі ме
   • Бұрын пайдаланылмаған ба
"""

import io
import os
import re
import json
import sqlite3
import datetime as dt
from pathlib import Path
from contextlib import contextmanager

import cv2
import httpx
import numpy as np
import pymupdf as fitz
from pdfminer.high_level import extract_text
from pdfminer.layout import LAParams
from dotenv import load_dotenv

from telegram import Update
from telegram.ext import (
    ApplicationBuilder, CommandHandler, MessageHandler,
    filters, ContextTypes,
)


# ============================================================
# CONFIG
# ============================================================
load_dotenv()

TOKEN: str = os.getenv("TG_BOT_TOKEN", "")
ADMIN_IDS: list[int] = [
    int(x.strip()) for x in (os.getenv("ADMIN_IDS") or "").split(",") if x.strip()
]
DB_PATH = Path("data/bot.db")
DB_PATH.parent.mkdir(parents=True, exist_ok=True)

if not TOKEN:
    raise RuntimeError("TG_BOT_TOKEN жоқ (.env файлын толтырыңыз)")


# ============================================================
# STORAGE
# ============================================================
SCHEMA = """
CREATE TABLE IF NOT EXISTS settings (
    key TEXT PRIMARY KEY,
    value TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS receipts (
    receipt_no TEXT PRIMARY KEY,
    user_id INTEGER NOT NULL,
    amount INTEGER NOT NULL,
    company_bin TEXT,
    used_at TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS users (
    user_id INTEGER PRIMARY KEY,
    registered_at TEXT NOT NULL
);
"""


@contextmanager
def db():
    conn = sqlite3.connect(DB_PATH, timeout=30)
    conn.row_factory = sqlite3.Row
    try:
        yield conn
        conn.commit()
    except Exception:
        conn.rollback()
        raise
    finally:
        conn.close()


def init_db():
    with db() as c:
        c.executescript(SCHEMA)


def get_setting(key: str):
    with db() as c:
        r = c.execute("SELECT value FROM settings WHERE key=?", (key,)).fetchone()
    return r["value"] if r else None


def set_setting(key: str, value: str):
    with db() as c:
        c.execute("INSERT OR REPLACE INTO settings VALUES (?, ?)", (key, value))


def register_user(uid: int):
    with db() as c:
        c.execute("INSERT OR IGNORE INTO users VALUES (?, ?)",
                  (uid, dt.datetime.now().isoformat()))


def receipt_used(rno: str) -> bool:
    with db() as c:
        return c.execute(
            "SELECT 1 FROM receipts WHERE receipt_no=?", (rno,)
        ).fetchone() is not None


def save_receipt(uid: int, rno: str, amount: int, bin_: str) -> bool:
    try:
        with db() as c:
            c.execute("INSERT INTO receipts VALUES (?, ?, ?, ?, ?)",
                      (rno, uid, amount, bin_, dt.datetime.now().isoformat()))
        return True
    except sqlite3.IntegrityError:
        return False


# ============================================================
# KASPI VERIFIER
# ============================================================
QR_LINK_RE = re.compile(r"https://receipt\.kaspi\.kz/web/fiscal\?[^\s\"'<>]+")
HTTP_HEADERS = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
                  "AppleWebKit/537.36 (KHTML, like Gecko) Chrome/156.0.0.0 Safari/537.36",
    "Accept-Language": "ru-RU,ru;q=0.9",
}


def pdf_text(pdf_bytes: bytes) -> str:
    parts = []
    try:
        t = extract_text(io.BytesIO(pdf_bytes), laparams=LAParams()) or ""
        if t.strip():
            parts.append(t)
    except Exception:
        pass
    try:
        doc = fitz.open(stream=pdf_bytes, filetype="pdf")
        pts = [p.get_text("text") for p in doc if p.get_text("text").strip()]
        if pts:
            parts.append("\n".join(pts))
    except Exception:
        pass
    return "\n\n".join(parts)


def pdf_qr(pdf_bytes: bytes) -> str | None:
    try:
        doc = fitz.open(stream=pdf_bytes, filetype="pdf")
    except Exception:
        return None
    det = cv2.QRCodeDetector()
    for page in doc:
        for info in page.get_images(full=True):
            try:
                pix = fitz.Pixmap(doc, info[0])
                if pix.n - pix.alpha > 3:
                    pix = fitz.Pixmap(fitz.csRGB, pix)
                img = np.frombuffer(pix.samples, dtype=np.uint8).reshape(
                    pix.height, pix.width, pix.n
                )
                if pix.n == 1:
                    img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
                elif pix.n == 4:
                    img = cv2.cvtColor(img, cv2.COLOR_RGBA2BGR)
                data, _, _ = det.detectAndDecode(img)
                if data and "kaspi" in data.lower():
                    return data.strip()
            except Exception:
                continue
        try:
            pix = page.get_pixmap(dpi=300)
            img = np.frombuffer(pix.samples, dtype=np.uint8).reshape(
                pix.height, pix.width, pix.n
            )
            if pix.n == 4:
                img = cv2.cvtColor(img, cv2.COLOR_RGBA2BGR)
            elif pix.n == 1:
                img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
            data, _, _ = det.detectAndDecode(img)
            if data and "kaspi" in data.lower():
                return data.strip()
        except Exception:
            continue
    return None


def _resolve_nuxt(data: list, ref, depth: int = 0, inside: bool = False):
    if depth > 40:
        return None
    if inside and isinstance(ref, int) and not isinstance(ref, bool):
        return _resolve_nuxt(data, data[ref], depth + 1, False) if 0 <= ref < len(data) else None
    if isinstance(ref, dict):
        return {k: _resolve_nuxt(data, v, depth + 1, True) for k, v in ref.items()}
    if isinstance(ref, list):
        return [_resolve_nuxt(data, v, depth + 1, True) for v in ref]
    return ref


def kaspi_fetch(url: str) -> dict:
    with httpx.Client(timeout=20, follow_redirects=True, headers=HTTP_HEADERS) as c:
        r = c.get(url)
        r.raise_for_status()
        html = r.text
    m = re.search(
        r'<script type="application/json" id="__NUXT_DATA__"[^>]*>(.*?)</script>',
        html, re.DOTALL,
    )
    if not m:
        raise ValueError("Kaspi беті өзгерген (__NUXT_DATA__ жоқ)")
    data = json.loads(m.group(1))
    idx = next(
        (i for i, it in enumerate(data)
         if isinstance(it, dict) and "amount" in it and "extTranId" in it),
        None,
    )
    if idx is None:
        raise ValueError("Чек объектісі табылмады")
    r = _resolve_nuxt(data, data[idx], inside=False) or {}
    params = {}
    for p in r.get("payParameters") or []:
        if isinstance(p, dict) and "name" in p and "value" in p:
            params[str(p["name"])] = p["value"]
    return {
        "amount": r.get("amount"),
        "receipt_no": r.get("extTranId"),
        "receipt_link": r.get("receiptLink"),
        "params": params,
    }


def find_qr_url(pdf_bytes: bytes) -> tuple[str | None, str]:
    text = pdf_text(pdf_bytes)
    m = QR_LINK_RE.search(text)
    if m:
        return m.group(0), text
    return pdf_qr(pdf_bytes), text


def extract_seller_info(pdf_bytes: bytes) -> dict:
    """PDF/Kaspi чегінен сатушының атауы мен ЖСН/БСН-ын алу."""
    text = pdf_text(pdf_bytes)
    company = None
    company_bin = None

    for pattern in (
        r"(?im)\bИП\s+([^\r\n]{2,80})",
        r"(?im)\bИП\s*[:：]\s*([^\r\n]{2,80})",
    ):
        m = re.search(pattern, text)
        if m:
            cand = re.split(r"\s{2,}|Сату|Түбіртек|Чек", m.group(1), maxsplit=1)[0]
            cand = cand.strip(" .,:;—-")
            if cand and len(cand) <= 80:
                company = cand
                break

    m = re.search(
        r"(?i)(?:ИИН\s*/\s*БИН|ЖСН\s*/?\s*БСН|БИН|ЖСН|ИИН)[^0-9]{0,30}(\d{12})",
        text,
    )
    if m:
        company_bin = m.group(1)

    qr_url, _ = find_qr_url(pdf_bytes)
    if qr_url and (not company or not company_bin):
        try:
            kaspi = kaspi_fetch(qr_url)
            params = kaspi.get("params") or {}
            if not company_bin:
                company_bin = str(
                    params.get("ИИН/БИН продавца", "") or ""
                ).strip() or None
            if not company:
                for label in ("Наименование продавца", "Продавец", "ИП"):
                    v = params.get(label)
                    if v:
                        company = str(v).strip()
                        break
        except Exception:
            pass

    if company_bin and not re.fullmatch(r"\d{12}", company_bin):
        company_bin = None
    return {"company": company, "company_bin": company_bin}


# ============================================================
# HANDLERS
# ============================================================
async def cmd_start(update: Update, context: ContextTypes.DEFAULT_TYPE):
    uid = update.effective_user.id
    register_user(uid)

    raw = get_setting("seller")
    if raw:
        s = json.loads(raw)
        await update.message.reply_text(
            "🤖 *Kaspi чек тексеру боты*\n\n"
            f"🏢 Сақталған компания: *{s['company']}*\n"
            f"🔢 ЖСН/БСН: `{s['company_bin']}`\n\n"
            "📄 PDF чекті жіберіңіз — бот тексереді:\n"
            "• Чек Kaspi-де түпнұсқа ма\n"
            "• Сол компанияға тиесілі ме\n"
            "• Бұрын пайдаланылмаған ба",
            parse_mode="Markdown",
        )
    else:
        await update.message.reply_text(
            "🤖 *Kaspi чек тексеру боты*\n\n"
            "📄 *Алғашқы* PDF чекті жіберіңіз — бот компанияны есте сақтайды.\n"
            "Кейін әр чек осы компанияға тексеріледі.",
            parse_mode="Markdown",
        )


async def cmd_status(update: Update, context: ContextTypes.DEFAULT_TYPE):
    raw = get_setting("seller")
    if not raw:
        await update.message.reply_text("ℹ️ Компания әлі сақталмаған. Бірінші чекті жіберіңіз.")
        return
    s = json.loads(raw)
    await update.message.reply_text(
        "🏢 *Сақталған компания*\n\n"
        f"Атауы: *{s['company']}*\n"
        f"ЖСН/БСН: `{s['company_bin']}`\n"
        f"Үлгі чек: `{s.get('sample_receipt') or '—'}`",
        parse_mode="Markdown",
    )


async def cmd_reset(update: Update, context: ContextTypes.DEFAULT_TYPE):
    uid = update.effective_user.id
    if uid not in ADMIN_IDS:
        await update.message.reply_text("❌ Тек админ үшін.")
        return
    with db() as c:
        c.execute("DELETE FROM settings WHERE key='seller'")
    await update.message.reply_text(
        "✅ Компания тазаланды. Келесі чек компанияны қайта орнатады."
    )


async def on_document(update: Update, context: ContextTypes.DEFAULT_TYPE):
    doc = update.message.document
    if not doc:
        return
    fname = (doc.file_name or "").lower()
    if not (doc.mime_type == "application/pdf" or fname.endswith(".pdf")):
        await update.message.reply_text("❌ Тек PDF чек жіберіңіз.")
        return

    uid = update.effective_user.id
    register_user(uid)

    msg = await update.message.reply_text("🔎 Чек тексерілуде...")

    try:
        f = await doc.get_file()
        pdf_bytes = bytes(await f.download_as_bytearray())
    except Exception as e:
        await msg.edit_text(f"❌ Файл жүктелмеді: {e}")
        return

    if len(pdf_bytes) > 10 * 1024 * 1024:
        await msg.edit_text("❌ Файл тым үлкен (10 MB).")
        return

    raw = get_setting("seller")
    if not raw:
        await _save_company(update, msg, pdf_bytes)
    else:
        await _verify_receipt(update, msg, pdf_bytes, json.loads(raw))


async def _save_company(update: Update, msg, pdf_bytes: bytes):
    """Бірінші чек: компанияны есте сақтау."""
    info = extract_seller_info(pdf_bytes)
    if not info.get("company_bin"):
        await msg.edit_text(
            "❌ Чектен сатушының 12 таңбалы ЖСН/БСН-ы табылмады.\n"
            "Басқа чек жіберіңіз."
        )
        return

    qr_url, _ = find_qr_url(pdf_bytes)
    if not qr_url:
        await msg.edit_text(
            "❌ Чекте QR-код немесе Kaspi сілтемесі табылмады.\n"
            "Бұл Kaspi фискалдық чек болуы керек."
        )
        return

    try:
        kaspi = kaspi_fetch(qr_url)
    except Exception as e:
        await msg.edit_text(f"❌ Kaspi-ден деректер алынбады: {e}")
        return

    data = {
        "company": info.get("company") or "Белгісіз сатушы",
        "company_bin": info["company_bin"],
        "sample_receipt": kaspi.get("receipt_no"),
    }
    set_setting("seller", json.dumps(data, ensure_ascii=False))

    # Осы чекті де "пайдаланылған" деп белгілейміз
    if kaspi.get("receipt_no"):
        save_receipt(
            update.effective_user.id,
            kaspi["receipt_no"],
            int(kaspi.get("amount") or 0),
            data["company_bin"],
        )

    await msg.edit_text(
        "✅ *Компания есте сақталды*\n\n"
        f"🏢 {data['company']}\n"
        f"🔢 `{data['company_bin']}`\n"
        f"📄 Үлгі чек: `{data['sample_receipt'] or '—'}`\n\n"
        "Енді кез келген чек осы компанияға тексеріледі.",
        parse_mode="Markdown",
    )


async def _verify_receipt(update: Update, msg, pdf_bytes: bytes, seller: dict):
    """Кейінгі чек: түпнұсқалық + компания + қайталамау."""
    qr_url, text = find_qr_url(pdf_bytes)
    if not qr_url:
        await msg.edit_text("❌ Чекте QR-код табылмады.")
        return

    try:
        kaspi = kaspi_fetch(qr_url)
    except Exception as e:
        await msg.edit_text(f"❌ Kaspi-ден деректер алынбады: {e}")
        return

    params = kaspi.get("params") or {}
    rno = kaspi.get("receipt_no")
    amount = int(kaspi.get("amount") or 0)
    k_bin = str(params.get("ИИН/БИН продавца", "")).strip()

    checks = []

    # 1. Түпнұсқалық (Kaspi базасында бар)
    checks.append(("✅ Чек Kaspi-де тіркелген", True, rno or "—"))

    # 2. Чек нөмірі PDF мәтінінде бар ма (PDF түпнұсқа екенін растайды)
    if rno:
        in_pdf = rno in (text or "")
        checks.append((
            "✅ Чек № PDF-те бар" if in_pdf else "❌ Чек № PDF-те жоқ",
            in_pdf,
            rno,
        ))

    # 3. Компания сәйкестігі
    match = (k_bin == seller["company_bin"])
    checks.append((
        "✅ Компания сәйкес" if match else "❌ Басқа компания",
        match,
        f"алынды `{k_bin or '—'}`, күтілді `{seller['company_bin']}`",
    ))

    # 4. Бұрын пайдаланылмаған
    used = receipt_used(rno) if rno else False
    checks.append((
        "❌ Чек бұрын пайдаланылған" if used else "✅ Чек жаңа",
        not used,
        rno or "—",
    ))

    all_ok = all(c[1] for c in checks)

    if all_ok and rno:
        save_receipt(update.effective_user.id, rno, amount, seller["company_bin"])

    lines = ["*🔎 Тексеру нәтижесі*\n"]
    for label, _ok, detail in checks:
        lines.append(f"{label}")
        if detail:
            lines.append(f"     `{detail}`")

    lines.append("")
    if all_ok:
        lines.append(
            f"✅ *Чек жарамды*\n"
            f"💳 Сома: *{amount:,} ₸*\n"
            f"🏢 Компания: *{seller['company']}*"
        )
    else:
        lines.append("❌ *Чек жарамсыз*")

    await msg.edit_text("\n".join(lines), parse_mode="Markdown")


# ============================================================
# MAIN
# ============================================================
def main():
    init_db()
    app = (
        ApplicationBuilder()
        .token(TOKEN)
        .build()
    )

    app.add_handler(CommandHandler("start", cmd_start))
    app.add_handler(CommandHandler("status", cmd_status))
    app.add_handler(CommandHandler("reset", cmd_reset))
    app.add_handler(MessageHandler(filters.Document.ALL, on_document))

    print("🚀 Бот іске қосылды")
    app.run_polling(drop_pending_updates=True)


if __name__ == "__main__":
    main()