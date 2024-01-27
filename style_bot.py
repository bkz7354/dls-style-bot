import logging
import os
import io
import re
import asyncio
from concurrent.futures import ProcessPoolExecutor
from multiprocessing import get_context

import dotenv
from telegram.ext._utils.types import FilterDataDict
import torch
import torchvision.transforms as transforms
from stylegan.networks import Generator
from PIL import Image


from telegram import Message, Update
from telegram.ext import ( 
    filters, MessageHandler, ApplicationBuilder, ContextTypes, 
    CommandHandler, ConversationHandler
)


welcome_message = """
Welcome to styleBot. Type /style_transfer to begin. Type /help for more info.
"""
async def start_command(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """
    Handle /start command
    """
    await context.bot.send_message(update.effective_chat.id, welcome_message)

help_message = """
Type /style_transfer to begin. The bot will ask you to send relevant photos.\
Note that bot ignores files and albums. At any time you can type /cancel to\
exit the conversation.
"""
async def help_command(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """
    Handle /help command
    """
    await context.bot.send_message(update.effective_chat.id, help_message)


async def style_no_photo(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    await update.effective_chat.send_message(
        "Please attach the photo you want to change when you send the command."
    )

class StyleFilter(filters.MessageFilter):
    def __init__(self):
        super().__init__()

        self.pattern = re.compile("^/style_transfer")
    
    def filter(self, message: Message) -> bool | FilterDataDict | None:
        if message.caption:
            return bool(re.search(self.pattern, message.caption))
        return False

def handle_transfer(image: Image, style: str, device: torch.device, imsize: int):
    return image

AVAILABLE_STYLES = ['monet', 'vangogh']
async def style_command(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    caption_words = update.message.caption.split()
    if len(caption_words) < 2:
        await update.effective_chat.send_message('You need to specify a style.')
        return
    style = caption_words[1]
    if style not in AVAILABLE_STYLES:
        await update.effective_chat.send_message('Available styles: {}.'.format(', '.join(AVAILABLE_STYLES)))
        return

    await update.effective_chat.send_message('Processing your image.')

    photo_id = update.message.photo[-1].file_id
    photo = await context.bot.get_file(photo_id)
    file = await photo.download_as_bytearray()

    image = Image.open(io.BytesIO(file))

    logging.info('Handling style transfer for chat %s.', update.effective_chat.id)
    loop = asyncio.get_running_loop()
    result_img = await loop.run_in_executor(PROC_POOL,
                                            handle_transfer, image, style, TORCH_DEVICE, TORCH_IMSIZE)

    await update.effective_chat.send_photo(PIL_to_bytes(result_img))




TORCH_DEVICE = torch.device('cpu')
TORCH_IMSIZE = 128
PROC_POOL = ProcessPoolExecutor()
def setup_pytorch(max_workers: int = 1) -> None:
    global TORCH_DEVICE
    global TORCH_IMSIZE
    global PROC_POOL
    
    TORCH_DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    TORCH_IMSIZE = 256 if torch.cuda.is_available() else 128
    torch.set_default_device(TORCH_DEVICE)

    logging.info("Torch device is %s, image size for processing set to %i", TORCH_DEVICE, TORCH_IMSIZE)

    mp_context = get_context('spawn')
    PROC_POOL = ProcessPoolExecutor(max_workers=max_workers, mp_context=mp_context)
    
def PIL_to_bytes(image: Image) -> io.BytesIO:
    bio = io.BytesIO()

    image.save(bio, 'JPEG')
    bio.seek(0)

    return bio

def parse_log_level(var_name: str, default_level: int) -> int:
    logging_levels = [logging.DEBUG, logging.INFO, logging.ERROR,  logging.CRITICAL]

    try:
        level = int(os.environ[var_name])
        if level not in logging_levels:
            raise RuntimeError("Unknown logging level")
        return level
    except KeyError:
        print(f"Logging level {var_name} not provided. Defaulting to {default_level}.")
    except ValueError:
        print(f"Cannot parse logging level {var_name}. Defaulting to {default_level}.")
    except RuntimeError:
        print(f"Unknown logging level {var_name}. Defaulting to {default_level}.")

    return default_level
    

def setup_logging() -> None:
    bot_level = parse_log_level('TG_BOT_LOG_LEVEL', logging.INFO)

    basic_conf_args = {
        'format': '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        'level': bot_level
    }
    if 'TG_BOT_LOG_FILE' in os.environ:
        basic_conf_args['filename'] = os.environ['TG_BOT_FILE']
    logging.basicConfig(**basic_conf_args)

    http_level = parse_log_level('HTTPX_LOG_LEVEL', logging.WARNING)
    logging.getLogger("httpx").setLevel(http_level)


def main() -> None:
    print("Loading environment variables from .env")
    dotenv.load_dotenv('.env')
    if 'TG_BOT_TOKEN' not in os.environ:
        print("Please provide a TG_BOT_TOKEN environment variable")
        return

    setup_logging()
    setup_pytorch()


    bot_token = os.environ['TG_BOT_TOKEN']
    application = ApplicationBuilder().token(bot_token).build()


    start_handler = CommandHandler('start', start_command)
    help_handler = CommandHandler('help', help_command)

    no_photo = CommandHandler('style_transfer', style_no_photo)
    command_filter = StyleFilter()
    style_handler = MessageHandler(filters.PHOTO & command_filter, style_command)

    application.add_handler(start_handler)
    application.add_handler(help_handler)
    application.add_handler(style_handler)
    application.add_handler(no_photo)

    try:
        application.run_polling()
    except Exception as e:
        print(e)

if __name__ == "__main__":
    main()
