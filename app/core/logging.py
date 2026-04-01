import logging


def setup_logging(log_level: str) -> None:
    level_name = log_level.upper()
    level = getattr(logging, level_name, logging.INFO)

    root_logger = logging.getLogger()
    if not root_logger.handlers:
        logging.basicConfig(
            level=level,
            format="%(asctime)s %(levelname)s %(name)s %(message)s",
        )
    else:
        root_logger.setLevel(level)


def get_logger(name: str) -> logging.Logger:
    return logging.getLogger(name)
