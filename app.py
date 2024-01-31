import logging
import os
import io
import re
import asyncio
from concurrent.futures import ProcessPoolExecutor
from multiprocessing import get_context
from typing import List
from textwrap import dedent

import dotenv
from telegram.ext._utils.types import FilterDataDict
import torch
import torchvision.transforms as transforms
from cyclegan.networks import Generator, denormalize
from PIL import Image


from telegram import Message, Update
from telegram.ext import ( 
    filters, MessageHandler, ApplicationBuilder, ContextTypes, 
    CommandHandler
)


async def start_command(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    await context.bot.send_message(
        update.effective_chat.id, 
        "Welcome to GANStyleBot. Type /help for more info."
    )

async def help_command(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    help_message = dedent(
        f"""
        Usage:
        /transfer_style {{{','.join(context.bot_data['styles'])}}}:
        You also need to attach the photo you want to change.
        """
    ).strip('\n')
    await context.bot.send_message(update.effective_chat.id, help_message)


async def style_command_no_photo(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    # this handler is called when user sends the command without attaching the photo
    await update.effective_chat.send_message(
        "Please attach the photo you want to change when you send the command."
    )

class StyleFilter(filters.MessageFilter):
    def __init__(self):
        super().__init__()

        self.pattern = re.compile("^/transfer_style")
    
    def filter(self, message: Message) -> bool | FilterDataDict | None:
        if message.caption:
            return bool(re.search(self.pattern, message.caption))
        return False


def handle_transfer(image: Image, style: str, device: torch.device):
    torch.set_default_device(device)
    saved_size = image.size

    model = Generator()
    model.load_state_dict(torch.load(f'./cyclegan/{style}_style.pth', map_location=device))
    model.to(device)

    input_transform = transforms.Compose([
        transforms.ToTensor(),
        # transforms.Resize(size=256, antialias=True),
        transforms.Normalize(
            mean=torch.tensor([0.5, 0.5, 0.5]),
            std=torch.tensor([0.5, 0.5, 0.5])
        )
    ])

    image = input_transform(image).to(device)
    result = denormalize(model(image)).cpu().detach()
    result = transforms.functional.to_pil_image(result)

    return result.resize(saved_size)


async def style_command(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    caption_words = update.message.caption.split()
    if len(caption_words) < 2:
        # user did not specify a style
        await update.effective_chat.send_message('You need to specify a style.')
        return
    
    style = caption_words[1]
    if style not in context.bot_data['styles']:
        # user specified an unknown style
        await update.effective_chat.send_message(
            'Available styles: {}.'.format(', '.join(context.bot_data['styles']))
        )
        return

    await update.effective_chat.send_message('Processing your image.')

    try:
        # get a suiltable image size
        photo_id = next(
            size.file_id for size in reversed(update.message.photo)
                if size.width <= context.bot_data['max_imsize'] and 
                    size.height <= context.bot_data['max_imsize'] 
        )
    except StopIteration:
        # no suitable image size found
        await update.effective_chat.send_message('Your image is too big')
        return
    
    # get the image
    photo = await context.bot.get_file(photo_id)
    file = await photo.download_as_bytearray()
    image = Image.open(io.BytesIO(file))

    # run style tranfer asynchronously
    logging.info('Handling style transfer for chat %s.', update.effective_chat.id)
    loop = asyncio.get_running_loop()
    result_img = await loop.run_in_executor(context.bot_data['proc_pool'],
                                            handle_transfer, image, style, context.bot_data['device'])

    await update.effective_chat.send_photo(PIL_to_bytes(result_img))



def get_available_styles(folder: str) -> List[str]:
    # scans the given folder for files named "{style_name}_style.pth"
    pattern = re.compile("([a-z]*)_style.pth$")

    result = []
    for item in os.listdir(folder):
        match = pattern.match(item)
        if match:
            result.append(match[1])

    return result

def get_env_int(env_name: str, default: int):
    try:
        return int(os.environ[env_name])
    except (KeyError, ValueError):
        return default

def setup_pytorch(bot_data: dict) -> None:
    # write some globals into bot_data

    bot_data['device'] = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.set_default_device(bot_data['device'])
    logging.info("Torch device is %s", bot_data['device'].type)

    # style transfer is blocking, so it will be run in a process pool 
    mp_context = get_context('spawn')
    max_workers = get_env_int('TG_BOT_MAX_WORKERS', 3)
    
    bot_data['proc_pool'] = ProcessPoolExecutor(max_workers=max_workers, mp_context=mp_context)
    logging.info("Pool workers set to %i", max_workers)

    bot_data['styles'] = get_available_styles('./cyclegan')

    max_imsize = get_env_int('TG_BOT_MAX_IMSIZE', 1024)
    bot_data['max_imsize'] = max_imsize

    
def PIL_to_bytes(image: Image) -> io.BytesIO:
    # convert PIL image to bytesIO to send it to user
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
        basic_conf_args['filename'] = os.environ['TG_BOT_LOG_FILE']
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

    bot_token = os.environ['TG_BOT_TOKEN']
    application = ApplicationBuilder().token(bot_token).build()

    setup_pytorch(application.bot_data)

    start_handler = CommandHandler('start', start_command)
    help_handler = CommandHandler('help', help_command)

    # handlers for style transfer
    no_photo = CommandHandler('transfer_style', style_command_no_photo)
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
