"""
Configure logging for the library.
"""
import logging


def defaultLogSetup():
    """
    Setup a simple logger.
    """
    logging.basicConfig(format='%(levelname)s:%(name)s: %(message)s')


def getLog(name, level=logging.DEBUG):
    """ Returns a logger object with the specified name and severity level.
    """
    log = logging.getLogger(name)
    log.setLevel(level)
    return log
