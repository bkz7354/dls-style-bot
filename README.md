## Style bot for DLS
This telegram bot uses StyleGAN to transfer styles from paintings to photos (available styles are Monet and Vangogh). 
It is built using [Pytorch](https://pytorch.org/)
and [python-telegram-bot](https://github.com/python-telegram-bot/python-telegram-bot) libraries.
Original realization of CycleGAN architecture
and the relevant paper can be found [here](https://junyanz.github.io/CycleGAN/). I wrote my own realization and used datasets from the original paper to train it. The training notebooks and
generated sample images can be found in a
[separate repository](https://example.com).

## Installation and configuration

The bot runs inference asynchronously, therefore it can be reasonably run without cuda. A separate 
`requirements_no_cuda.txt` file is provided to run
bot on cpu.

#### Install cpu-only
```
git clone TBD
cd style_bot
python3 -m pip install -r requirements_no_cuda.txt --extra-index-url https://download.pytorch.org/whl/cpu
```
#### Install with cuda
```
git clone TBD
cd style_bot
python3 -m pip install -r requirements.txt
```


To run the bot you need to provide an API token using `TG_BOT_TOKEN` environment variable.
You can also use a `.env` file. There are also other configuration options available:
- `TG_BOT_LOG_LEVEL` - overall logging level of the bot. Default is `logging.INFO`.
- `HTTPX_LOG_LEVEL` - logging level for httpx library used by PTB. Default is `logging.WARNING`.

> [!NOTE] Bot expects a standart python logging level in numeric format. See more [here](https://docs.python.org/3/library/logging.html#logging-levels).

- `TG_BOT_LOG_FILE` - the bot will write logs into the given file if the variable is set.
- `TG_BOT_MAX_WORKERS` - the max_workers argument passed to [`ProcessPoolExecutor`](https://docs.python.org/3/library/concurrent.futures.html#processpoolexecutor). It limits inferences that can be run concurrently. Default value is 3.
- `TG_BOT_MAX_IMSIZE` - limits width and height of images which bot will process. Default is 1024.

> [!NOTE] Since Telegram provides several image sizes to choose from, the bot will not refuse
huge images if a smaller resolution image can be processed.

## Usage

The bot can be run with
```
python3 app.py 
```
To do the inference `/transfer_style {monet,vangogh}` command is used. The photo needs to be attached to the command when it is sent. Example usage:


