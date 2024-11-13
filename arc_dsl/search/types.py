from dataclasses import dataclass, field

from arc_dsl.interpreter import Symbol


@dataclass
class Pattern:
    symbol: Symbol
    arg_classes: list = field(default_factory=list)
    type: any = None

    @property
    def arity(self):
        return len(self.arg_classes)


@dataclass
class ArgPattern:
    func: bool = False
    arity: int = -1
    type: any = None
