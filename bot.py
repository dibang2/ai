import os
import json
import html
import logging
import asyncio
import threading
from typing import Dict, List

from openai import AsyncOpenAI
from dotenv import load_dotenv
from telegram import Update, InlineKeyboardButton, InlineKeyboardMarkup
from telegram.ext import (
    Application,
    CommandHandler,
    MessageHandler,
    CallbackQueryHandler,
    filters,
    ContextTypes,
)

load_dotenv()

logging.basicConfig(
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    level=logging.INFO,
)
logger = logging.getLogger(__name__)

TELEGRAM_TOKEN = os.getenv("TELEGRAM_TOKEN")
SYSTEM_PROMPT = "You are a helpful AI assistant."
API_BASE_URL = "https://integrate.api.nvidia.com/v1"
DEFAULT_QUOTA = 100
DATA_DIR = "data"
_SAVE_INTERVAL = 60  # auto-save every 60 seconds
_save_timer: threading.Timer | None = None

MODELS = {
    "glm4.7": {
        "label": "GLM-4.7",
        "api_key": "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",
        "model": "z-ai/glm4.7",
        "temperature": 1.0,
        "top_p": 1.0,
        "max_tokens": 16384,
        "extra_body": {
            "chat_template_kwargs": {
                "enable_thinking": True,
                "clear_thinking": False,
            }
        },
    },
    "deepseek-v4-flash": {
        "label": "DeepSeek V4 Flash",
        "api_key": "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",
        "model": "deepseek-ai/deepseek-v4-flash",
        "temperature": 1.0,
        "top_p": 0.95,
        "max_tokens": 16384,
        "extra_body": {
            "chat_template_kwargs": {
                "thinking": True,
                "reasoning_effort": "high",
            }
        },
    },
    "deepseek-v4-pro": {
        "label": "DeepSeek V4 Pro",
        "api_key": "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",
        "model": "deepseek-ai/deepseek-v4-pro",
        "temperature": 1.0,
        "top_p": 0.95,
        "max_tokens": 16384,
        "extra_body": {
            "chat_template_kwargs": {
                "thinking": False,
            }
        },
    },
    "qwen-3-next": {
        "label": "Qwen3 Next 80B",
        "api_key": "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",
        "model": "qwen/qwen3-next-80b-a3b-instruct",
        "temperature": 0.6,
        "top_p": 0.7,
        "max_tokens": 4096,
        "extra_body": None,
    },
    "llama-3.3-70b": {
        "label": "Llama 3.3 70B",
        "api_key": "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",
        "model": "meta/llama-3.3-70b-instruct",
        "temperature": 0.2,
        "top_p": 0.7,
        "max_tokens": 1024,
        "extra_body": None,
    },
}

DEFAULT_MODEL = "deepseek-v4-flash"

users: Dict[int, dict] = {}
conversations: Dict[int, List[dict]] = {}
user_models: Dict[int, str] = {}
user_usage: Dict[int, dict] = {}


def _data_path(name: str) -> str:
    return os.path.join(DATA_DIR, f"{name}.json")


def _save_json(name: str, data):
    path = _data_path(name)
    os.makedirs(DATA_DIR, exist_ok=True)
    tmp = path + ".tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2, default=str)
    os.replace(tmp, path)


def _load_json(name: str, default=None):
    path = _data_path(name)
    if not os.path.exists(path):
        return default if default is not None else {}
    try:
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return default if default is not None else {}


def _save_all():
    _save_json("users", {str(k): v for k, v in users.items()})
    _save_json("conversations", {str(k): v for k, v in conversations.items()})
    _save_json("user_models", {str(k): v for k, v in user_models.items()})
    _save_json("user_usage", {str(k): v for k, v in user_usage.items()})
    logger.info("数据已自动保存")


def _load_all():
    raw_users = _load_json("users", {})
    for k, v in raw_users.items():
        users[int(k)] = v

    raw_convs = _load_json("conversations", {})
    for k, v in raw_convs.items():
        conversations[int(k)] = v

    raw_models = _load_json("user_models", {})
    for k, v in raw_models.items():
        user_models[int(k)] = v

    raw_usage = _load_json("user_usage", {})
    for k, v in raw_usage.items():
        user_usage[int(k)] = v

    logger.info(f"已加载 {len(users)} 个用户，{len(conversations)} 个对话")


def _start_auto_save():
    global _save_timer
    if _save_timer and _save_timer.is_alive():
        return
    _save_timer = threading.Timer(_SAVE_INTERVAL, _auto_save_loop)
    _save_timer.daemon = True
    _save_timer.start()


def _auto_save_loop():
    try:
        _save_all()
    except Exception as e:
        logger.error(f"自动保存失败：{e}")
    global _save_timer
    _save_timer = threading.Timer(_SAVE_INTERVAL, _auto_save_loop)
    _save_timer.daemon = True
    _save_timer.start()


def get_client(model_key: str) -> AsyncOpenAI:
    cfg = MODELS[model_key]
    return AsyncOpenAI(base_url=API_BASE_URL, api_key=cfg["api_key"])


def _build_message(reasoning: str, content: str) -> str:
    parts = []
    if reasoning:
        parts.append(f"<blockquote>{html.escape(reasoning)}</blockquote>")
    if content:
        parts.append(html.escape(content))
    return "\n\n".join(parts) if parts else "."


def _truncate(text: str, max_len: int = 4000) -> str:
    if len(text) <= max_len:
        return text
    return text[: max_len - 3] + "..."


def _get_quota(user_id: int) -> int:
    u = users.get(user_id)
    if not u:
        return 0
    if u.get("is_admin"):
        return -1
    return u.get("quota", 0)


def _deduct_quota(user_id: int) -> bool:
    u = users.get(user_id)
    if not u:
        return False
    if u.get("is_admin"):
        return True
    if u.get("quota", 0) <= 0:
        return False
    u["quota"] -= 1
    return True


async def register(update: Update, context: ContextTypes.DEFAULT_TYPE):
    user_id = update.effective_user.id
    name = update.effective_user.full_name

    if user_id in users:
        await update.message.reply_text(f"{name}，你已经注册过了。")
        return

    is_admin = len(users) == 0
    users[user_id] = {"quota": DEFAULT_QUOTA, "is_admin": is_admin, "name": name}
    conversations[user_id] = [{"role": "system", "content": SYSTEM_PROMPT}]
    _save_all()

    msg = f"欢迎 {name}！你获得了 {DEFAULT_QUOTA} 次聊天额度。"
    if is_admin:
        msg += "\n\n你是**管理员**，拥有无限使用额度。发送 /admin 查看管理功能。"
    await update.message.reply_text(msg)


async def admin_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    user_id = update.effective_user.id
    u = users.get(user_id)
    if not u or not u.get("is_admin"):
        await update.message.reply_text("仅管理员可用。")
        return

    await update.message.reply_text(
        "**管理员命令：**\n\n"
        "/users - 查看所有用户及配额\n"
        "/setquota `<用户ID>` `<次数>` - 设置用户剩余次数\n"
        "/addquota `<用户ID>` `<次数>` - 增加用户次数\n\n"
        "使用 `-1` 表示无限额度。"
    )


async def users_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    user_id = update.effective_user.id
    u = users.get(user_id)
    if not u or not u.get("is_admin"):
        await update.message.reply_text("仅管理员可用。")
        return

    if not users:
        await update.message.reply_text("暂无注册用户。")
        return

    lines = ["**已注册用户：**\n"]
    for uid, info in users.items():
        quota = "∞" if info.get("is_admin") else info.get("quota", 0)
        role = "👑 管理员" if info.get("is_admin") else "用户"
        name = info.get("name", str(uid))
        lines.append(f"`{uid}` — {name} — {role} — 剩余次数：{quota}")

    await update.message.reply_text("\n".join(lines))


async def setquota_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    user_id = update.effective_user.id
    u = users.get(user_id)
    if not u or not u.get("is_admin"):
        await update.message.reply_text("仅管理员可用。")
        return

    args = context.args
    if len(args) < 2:
        await update.message.reply_text("用法：/setquota `<用户ID>` `<次数>`")
        return

    try:
        target_id = int(args[0])
        count = int(args[1])
    except ValueError:
        await update.message.reply_text("参数无效。")
        return

    if target_id not in users:
        await update.message.reply_text("未找到该用户。")
        return

    users[target_id]["quota"] = count
    _save_all()
    name = users[target_id].get("name", str(target_id))
    await update.message.reply_text(f"已将 {name} 的剩余次数设为 {count}。")


async def addquota_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    user_id = update.effective_user.id
    u = users.get(user_id)
    if not u or not u.get("is_admin"):
        await update.message.reply_text("仅管理员可用。")
        return

    args = context.args
    if len(args) < 2:
        await update.message.reply_text("用法：/addquota `<用户ID>` `<次数>`")
        return

    try:
        target_id = int(args[0])
        count = int(args[1])
    except ValueError:
        await update.message.reply_text("参数无效。")
        return

    if target_id not in users:
        await update.message.reply_text("未找到该用户。")
        return

    if not users[target_id].get("is_admin"):
        users[target_id]["quota"] = users[target_id].get("quota", 0) + count

    _save_all()
    name = users[target_id].get("name", str(target_id))
    await update.message.reply_text(f"已给 {name} 增加 {count} 次额度。当前：{users[target_id]['quota']}。")


async def help_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    user_id = update.effective_user.id
    model_key = user_models.get(user_id, DEFAULT_MODEL)
    model_label = MODELS[model_key]["label"]
    u = users.get(user_id)
    is_admin = u and u.get("is_admin")

    lines = [
        f"**AI 机器人帮助**\n\n"
        f"当前模型：{model_label}\n\n"
        "**命令列表：**\n"
        "/register - 注册使用机器人\n"
        "/start - 显示欢迎信息\n"
        "/models - 切换 AI 模型\n"
        "/clear - 清除对话历史\n"
        "/help - 显示此帮助\n"
        "/usage - 查看 Token 用量\n"
        "/me - 查看我的账户信息"
    ]

    if is_admin:
        lines.append("/admin - 管理员控制面板")

    await update.message.reply_text("\n".join(lines))


async def start(update: Update, context: ContextTypes.DEFAULT_TYPE):
    user_id = update.effective_user.id
    conversations[user_id] = [{"role": "system", "content": SYSTEM_PROMPT}]
    model_key = user_models.get(user_id, DEFAULT_MODEL)
    model_label = MODELS[model_key]["label"]

    u = users.get(user_id)
    if not u:
        await update.message.reply_text(
            f"你好！我是 AI 驱动的 Telegram 机器人。\n"
            f"当前模型：{model_label}\n\n"
            "你还没有注册，发送 /register 开始使用。"
        )
        return

    await update.message.reply_text(
        f"你好！我是 AI 驱动的 Telegram 机器人。\n"
        f"当前模型：{model_label}\n\n"
        "命令：\n"
        "/clear - 清除对话历史\n"
        "/models - 切换模型\n"
        "/help - 帮助\n"
        "/usage - Token 用量"
    )


async def me_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    user_id = update.effective_user.id
    u = users.get(user_id)
    if not u:
        await update.message.reply_text("未注册。请先发送 /register。")
        return

    role = "👑 管理员" if u.get("is_admin") else "用户"
    quota = "∞" if u.get("is_admin") else u.get("quota", 0)
    name = u.get("name", str(user_id))
    usage = user_usage.get(user_id, {})
    total_tokens = usage.get("total_tokens", 0)

    await update.message.reply_text(
        f"**我的账户**\n\n"
        f"名称：{name}\n"
        f"ID：`{user_id}`\n"
        f"角色：{role}\n"
        f"剩余次数：{quota}\n"
        f"累计 Token：{total_tokens}"
    )


async def clear(update: Update, context: ContextTypes.DEFAULT_TYPE):
    user_id = update.effective_user.id
    conversations[user_id] = [{"role": "system", "content": SYSTEM_PROMPT}]
    _save_all()
    await update.message.reply_text("对话历史已清除。")


async def models_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    user_id = update.effective_user.id
    current = user_models.get(user_id, DEFAULT_MODEL)
    keyboard = []
    for key, cfg in MODELS.items():
        prefix = "✅ " if key == current else ""
        keyboard.append(
            [InlineKeyboardButton(f"{prefix}{cfg['label']}", callback_data=f"model:{key}")]
        )
    await update.message.reply_text(
        "选择模型：", reply_markup=InlineKeyboardMarkup(keyboard)
    )


async def model_callback(update: Update, context: ContextTypes.DEFAULT_TYPE):
    query = update.callback_query
    await query.answer()
    user_id = query.from_user.id
    model_key = query.data.split(":", 1)[1]

    if model_key not in MODELS:
        await query.edit_message_text("未知模型。")
        return

    user_models[user_id] = model_key
    conversations[user_id] = [{"role": "system", "content": SYSTEM_PROMPT}]
    _save_all()
    model_label = MODELS[model_key]["label"]

    await query.edit_message_text(f"已切换到 {model_label}，对话历史已重置。")


async def usage_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    user_id = update.effective_user.id
    u = user_usage.get(user_id, {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0})
    await update.message.reply_text(
        f"📊 **Token 用量**\n\n"
        f"输入 Token：{u['prompt_tokens']}\n"
        f"输出 Token：{u['completion_tokens']}\n"
        f"总计 Token：{u['total_tokens']}"
    )


async def handle_message(update: Update, context: ContextTypes.DEFAULT_TYPE):
    user_id = update.effective_user.id
    user_message = update.message.text

    if user_id not in users:
        await update.message.reply_text(
            "你还没有注册。发送 /register 开始使用。"
        )
        return

    quota = _get_quota(user_id)
    if quota == 0:
        await update.message.reply_text(
            "你的额度已用完，请联系管理员补充。"
        )
        return

    model_key = user_models.get(user_id, DEFAULT_MODEL)

    if user_id not in conversations:
        conversations[user_id] = [{"role": "system", "content": SYSTEM_PROMPT}]

    conversations[user_id].append({"role": "user", "content": user_message})

    _deduct_quota(user_id)
    remaining = _get_quota(user_id)

    await update.message.chat.send_action(action="typing")

    try:
        cfg = MODELS[model_key]
        client = get_client(model_key)

        stream = await client.chat.completions.create(
            model=cfg["model"],
            messages=conversations[user_id],
            temperature=cfg["temperature"],
            top_p=cfg["top_p"],
            max_tokens=cfg["max_tokens"],
            stream=True,
            **(dict(extra_body=cfg["extra_body"]) if cfg.get("extra_body") else {}),
        )

        full_content = ""
        full_reasoning = ""
        message_obj = None
        last_edit_time = 0.0

        usage_data = {}

        async for chunk in stream:
            if not chunk.choices:
                continue
            delta = chunk.choices[0].delta
            if not delta:
                continue

            reasoning = (
                getattr(delta, "reasoning", None)
                or getattr(delta, "reasoning_content", None)
            )
            if reasoning:
                full_reasoning += reasoning

            if delta.content:
                full_content += delta.content

            if getattr(chunk, "usage", None):
                usage_data = {
                    "prompt_tokens": chunk.usage.prompt_tokens,
                    "completion_tokens": chunk.usage.completion_tokens,
                    "total_tokens": chunk.usage.total_tokens,
                }

            now = asyncio.get_event_loop().time()
            if message_obj is None and (full_content or full_reasoning):
                text = _truncate(_build_message(full_reasoning, full_content))
                message_obj = await update.message.reply_text(text, parse_mode="HTML")
                last_edit_time = now
            elif message_obj and now - last_edit_time >= 1.0:
                try:
                    text = _truncate(_build_message(full_reasoning, full_content))
                    await message_obj.edit_text(text, parse_mode="HTML")
                except Exception:
                    pass
                last_edit_time = now

        if message_obj:
            text = _truncate(_build_message(full_reasoning, full_content))
            try:
                await message_obj.edit_text(text, parse_mode="HTML")
            except Exception:
                pass
        else:
            text = full_content or "(空响应)"
            await update.message.reply_text(text)

        conversations[user_id].append(
            {"role": "assistant", "content": full_content}
        )
        _save_all()

        if usage_data:
            u = user_usage.setdefault(user_id, {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0})
            u["prompt_tokens"] += usage_data["prompt_tokens"]
            u["completion_tokens"] += usage_data["completion_tokens"]
            u["total_tokens"] += usage_data["total_tokens"]

            usage_text = (
                f"\n\n— Token：{usage_data['total_tokens']} "
                f"(↑{usage_data['prompt_tokens']} ↓{usage_data['completion_tokens']}) "
                f"| 累计：{u['total_tokens']}"
            )
            try:
                if message_obj:
                    final = _truncate(
                        _build_message(full_reasoning, full_content) + usage_text
                    )
                    await message_obj.edit_text(final, parse_mode="HTML")
            except Exception:
                pass

        if remaining >= 0 and remaining <= 5:
            warn = f"\n\n⚠️ 你还有 {remaining} 次聊天额度。"
            try:
                if message_obj:
                    final = _truncate(
                        _build_message(full_reasoning, full_content) + usage_text + warn
                    )
                    await message_obj.edit_text(final, parse_mode="HTML")
            except Exception:
                pass

    except Exception as e:
        err = str(e)
        if "image" in err.lower() and "not support" in err.lower():
            await update.message.reply_text("该模型不支持图片输入。")
        else:
            logger.error(f"API error: {e}")
            await update.message.reply_text(f"错误：{err}")


async def handle_non_text(update: Update, context: ContextTypes.DEFAULT_TYPE):
    user_id = update.effective_user.id
    if user_id not in users:
        return
    await update.message.reply_text("该模型不支持图片或文件输入，请发送文字消息。")


def main():
    if not TELEGRAM_TOKEN:
        logger.error("TELEGRAM_TOKEN 未设置！")
        return

    app = Application.builder().token(TELEGRAM_TOKEN).build()

    app.add_handler(CommandHandler("register", register))
    app.add_handler(CommandHandler("start", start))
    app.add_handler(CommandHandler("help", help_command))
    app.add_handler(CommandHandler("me", me_command))
    app.add_handler(CommandHandler("clear", clear))
    app.add_handler(CommandHandler("models", models_command))
    app.add_handler(CommandHandler("usage", usage_command))

    app.add_handler(CommandHandler("admin", admin_command))
    app.add_handler(CommandHandler("users", users_command))
    app.add_handler(CommandHandler("setquota", setquota_command))
    app.add_handler(CommandHandler("addquota", addquota_command))

    app.add_handler(CallbackQueryHandler(model_callback, pattern=r"^model:"))
    app.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, handle_message))
    app.add_handler(MessageHandler(~filters.TEXT & ~filters.COMMAND, handle_non_text))

    logger.info("机器人已启动，按 Ctrl+C 停止。")

    _load_all()
    _start_auto_save()

    app.run_polling(allowed_updates=Update.ALL_TYPES)


if __name__ == "__main__":
    main()