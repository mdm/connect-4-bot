import enum

class Player(enum.Enum):
    red = 1
    yellow = 2

    @property
    def other(self):
        return Player.red if self == Player.yellow else Player.yellow
