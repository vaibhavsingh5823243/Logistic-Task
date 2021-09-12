import logging as lg
class Log:
    def __init__(self,name):
        self.name=name
    def log(self):
        logger=lg.getLogger(self.name)
        logger.setLevel(lg.DEBUG)
        format=lg.Formatter("%(asctime)s:%(name)s:(levelname)s:%(message)s")
        filehandler=lg.FileHandler("test.log")
        filehandler.setFormatter(format)
        logger.addHandler(filehandler)
        return logger