# Code Conventions

## Class Member Ordering

Order class members top-to-bottom:

1. ClassVar/class-level constants
2. `__slots__`
3. `__init__`
4. Representation dunders (`__repr__`, `__str__`, `__eq__`, `__hash__`, `__len__`)
5. `@property` blocks (getter/setter/deleter together)
6. `@classmethod`
7. `@staticmethod`
8. Public instance methods
9. Protected methods (`_name`)
10. Private methods (`__name`)
11. Context-manager protocol (`__enter__/__exit__`, `__aenter__/__aexit__`)

Key rules:

- Group by visibility over decorator style
- Keep logically related methods adjacent within each section
- Put async before sync variants of the same operation
- Keep property getter/setter/deleter adjacent

Special cases:

- Pydantic `BaseModel`: field declarations first, validators next, regular methods after
- `@dataclass`: fields define class body; methods follow ordering above
- gRPC servicers: keep method order aligned with proto service definition

Enforcement note:

- Ruff has no class-member-order rule; enforce via code review
