"""
Expression engine for evaluating SQL-like calculation expressions.

Supports:
- Arithmetic: +, -, *, /
- Comparisons: >, <, >=, <=, ==, !=
- Logical operators: AND, OR, NOT
- CASE WHEN ... THEN ... ELSE ... END
- COALESCE(expr, expr, ...)
- DATEDIFF('unit', start, end)
- Column references: column_name or source.column_name
- Numeric and string literals
"""

import ast
import re
from datetime import datetime
from typing import Any, Dict, List, Optional


class ExpressionError(Exception):
    """Raised when an expression cannot be parsed or evaluated."""

    pass


def evaluate_expression(expr: str, row: Dict[str, Any]) -> Any:
    """
    Evaluate a SQL-like expression against a single row of data.

    Args:
        expr: The SQL-like expression string.
        row: A dictionary mapping column names to values.

    Returns:
        The computed result.

    Raises:
        ExpressionError: If the expression is invalid or evaluation fails.
    """
    try:
        tokens = tokenize(expr)
        parser = ExpressionParser(tokens, row)
        return parser.parse()
    except ExpressionError:
        raise
    except Exception as e:
        raise ExpressionError(f"Failed to evaluate expression '{expr}': {e}") from e


# ---------------------------------------------------------------------------
# Tokenizer
# ---------------------------------------------------------------------------

TOKEN_PATTERNS = [
    ("WHITESPACE", r"\s+"),
    ("NUMBER", r"\d+(?:\.\d+)?"),
    ("STRING", r"'(?:[^'\\]|\\.)*'"),
    ("LE", r"<="),
    ("GE", r">="),
    ("NE", r"!="),
    ("EQ", r"=="),
    ("LT", r"<"),
    ("GT", r">"),
    ("PLUS", r"\+"),
    ("MINUS", r"-"),
    ("MUL", r"\*"),
    ("DIV", r"/"),
    ("LPAREN", r"\("),
    ("RPAREN", r"\)"),
    ("COMMA", r","),
    ("DOT", r"\."),
    ("IDENT", r"[A-Za-z_][A-Za-z0-9_]*"),
]

_TOKEN_RE = re.compile("|".join(f"(?P<{name}>{pattern})" for name, pattern in TOKEN_PATTERNS))

# Keywords that map to specific token types
_KEYWORDS = {
    "AND",
    "OR",
    "NOT",
    "CASE",
    "WHEN",
    "THEN",
    "ELSE",
    "END",
    "COALESCE",
    "DATEDIFF",
    "TRUE",
    "FALSE",
    "NULL",
    "IS",
}


class Token:
    __slots__ = ("type", "value")

    def __init__(self, type: str, value: str):
        self.type = type
        self.value = value

    def __repr__(self) -> str:
        return f"Token({self.type}, {self.value!r})"


def tokenize(expr: str) -> List[Token]:
    """Tokenize an expression string into a list of Token objects."""
    tokens: List[Token] = []
    pos = 0
    while pos < len(expr):
        m = _TOKEN_RE.match(expr, pos)
        if m is None:
            raise ExpressionError(
                f"Unexpected character at position {pos}: {expr[pos]!r}"
            )
        kind = m.lastgroup
        value = m.group()
        pos = m.end()
        if kind == "WHITESPACE":
            continue
        if kind == "IDENT" and value.upper() in _KEYWORDS:
            tokens.append(Token(value.upper(), value))
        else:
            assert kind is not None
            tokens.append(Token(kind, value))
    tokens.append(Token("EOF", ""))
    return tokens


# ---------------------------------------------------------------------------
# Recursive-descent parser / evaluator
# ---------------------------------------------------------------------------

class ExpressionParser:
    """
    Recursive-descent parser that simultaneously evaluates the expression.

    Grammar (simplified):
        expr        -> or_expr
        or_expr     -> and_expr ( 'OR' and_expr )*
        and_expr    -> not_expr ( 'AND' not_expr )*
        not_expr    -> 'NOT' not_expr | comparison
        comparison  -> addition ( ( '==' | '!=' | '<' | '>' | '<=' | '>=' | 'IS' ) addition )?
        addition    -> multiply ( ( '+' | '-' ) multiply )*
        multiply    -> unary ( ( '*' | '/' ) unary )*
        unary       -> '-' unary | primary
        primary     -> NUMBER | STRING | TRUE | FALSE | NULL
                     | CASE ... END | COALESCE(...) | DATEDIFF(...)
                     | IDENT ( '.' IDENT )* | '(' expr ')'
    """

    def __init__(self, tokens: List[Token], row: Dict[str, Any]):
        self.tokens = tokens
        self.row = row
        self.pos = 0

    def parse(self) -> Any:
        result = self._expr()
        if self.tokens[self.pos].type != "EOF":
            raise ExpressionError(
                f"Unexpected token after expression: {self.tokens[self.pos]}"
            )
        return result

    # -- helpers --

    def _peek(self) -> Token:
        return self.tokens[self.pos]

    def _advance(self) -> Token:
        tok = self.tokens[self.pos]
        self.pos += 1
        return tok

    def _expect(self, token_type: str) -> Token:
        tok = self._advance()
        if tok.type != token_type and tok.value.upper() != token_type:
            raise ExpressionError(
                f"Expected {token_type}, got {tok}"
            )
        return tok

    def _match(self, *types: str) -> Optional[Token]:
        tok = self._peek()
        if tok.type in types or tok.value.upper() in types:
            return self._advance()
        return None

    # -- grammar rules --

    def _expr(self) -> Any:
        return self._or_expr()

    def _or_expr(self) -> Any:
        left = self._and_expr()
        while self._match("OR"):
            right = self._and_expr()
            left = bool(left) or bool(right)
        return left

    def _and_expr(self) -> Any:
        left = self._not_expr()
        while self._match("AND"):
            right = self._not_expr()
            left = bool(left) and bool(right)
        return left

    def _not_expr(self) -> Any:
        if self._match("NOT"):
            return not bool(self._not_expr())
        return self._comparison()

    def _comparison(self) -> Any:
        left = self._addition()

        # Handle IS NULL / IS NOT NULL
        if self._peek().value.upper() == "IS":
            self._advance()
            if self._peek().value.upper() == "NOT":
                self._advance()
                self._expect("NULL")
                return left is not None
            else:
                self._expect("NULL")
                return left is None

        op = self._match("EQ", "NE", "LT", "GT", "LE", "GE")
        if op:
            right = self._addition()
            if op.type == "EQ":
                return left == right
            elif op.type == "NE":
                return left != right
            elif op.type == "LT":
                return left < right
            elif op.type == "GT":
                return left > right
            elif op.type == "LE":
                return left <= right
            elif op.type == "GE":
                return left >= right
        return left

    def _addition(self) -> Any:
        left = self._multiply()
        while True:
            op = self._match("PLUS", "MINUS")
            if not op:
                break
            right = self._multiply()
            if op.type == "PLUS":
                left = left + right
            else:
                left = left - right
        return left

    def _multiply(self) -> Any:
        left = self._unary()
        while True:
            op = self._match("MUL", "DIV")
            if not op:
                break
            right = self._unary()
            if op.type == "MUL":
                left = left * right
            else:
                if right == 0:
                    raise ExpressionError("Division by zero")
                left = left / right
        return left

    def _unary(self) -> Any:
        if self._match("MINUS"):
            return -self._unary()
        return self._primary()

    def _primary(self) -> Any:
        tok = self._peek()

        # Numeric literal
        if tok.type == "NUMBER":
            self._advance()
            if "." in tok.value:
                return float(tok.value)
            return int(tok.value)

        # String literal
        if tok.type == "STRING":
            self._advance()
            return ast.literal_eval(tok.value)

        # Boolean literals
        if tok.value.upper() == "TRUE":
            self._advance()
            return True
        if tok.value.upper() == "FALSE":
            self._advance()
            return False

        # NULL
        if tok.value.upper() == "NULL":
            self._advance()
            return None

        # CASE expression
        if tok.value.upper() == "CASE":
            return self._case_expr()

        # COALESCE function
        if tok.value.upper() == "COALESCE":
            return self._coalesce_expr()

        # DATEDIFF function
        if tok.value.upper() == "DATEDIFF":
            return self._datediff_expr()

        # Parenthesized expression
        if tok.type == "LPAREN":
            self._advance()
            result = self._expr()
            self._expect("RPAREN")
            return result

        # Column reference: ident or ident.ident
        if tok.type == "IDENT":
            return self._column_ref()

        raise ExpressionError(f"Unexpected token: {tok}")

    def _column_ref(self) -> Any:
        """Parse a column reference like 'amount' or 'source.amount'."""
        parts = [self._advance().value]
        while self._match("DOT"):
            parts.append(self._expect("IDENT").value)

        # Try the full dotted name first, then just the last part
        full_name = ".".join(parts)
        underscore_name = "__".join(parts)

        if full_name in self.row:
            return self.row[full_name]
        if underscore_name in self.row:
            return self.row[underscore_name]
        if len(parts) > 1 and parts[-1] in self.row:
            return self.row[parts[-1]]
        if parts[0] in self.row:
            return self.row[parts[0]]

        raise ExpressionError(
            f"Column '{full_name}' not found. Available columns: {list(self.row.keys())}"
        )

    def _case_expr(self) -> Any:
        """Parse CASE WHEN ... THEN ... [WHEN ... THEN ...] [ELSE ...] END."""
        self._expect("CASE")
        result = None
        matched = False

        while self._peek().value.upper() == "WHEN":
            self._advance()  # consume WHEN
            condition = self._expr()
            self._expect("THEN")
            value = self._expr()
            if not matched and bool(condition):
                result = value
                matched = True

        if self._peek().value.upper() == "ELSE":
            self._advance()  # consume ELSE
            else_value = self._expr()
            if not matched:
                result = else_value

        self._expect("END")
        return result

    def _coalesce_expr(self) -> Any:
        """Parse COALESCE(expr, expr, ...)."""
        self._expect("COALESCE")
        self._expect("LPAREN")

        args = [self._expr()]
        while self._match("COMMA"):
            args.append(self._expr())
        self._expect("RPAREN")

        for arg in args:
            if arg is not None:
                return arg
        return None

    def _datediff_expr(self) -> Any:
        """Parse DATEDIFF('unit', start_expr, end_expr)."""
        self._expect("DATEDIFF")
        self._expect("LPAREN")

        unit_tok = self._expr()
        if not isinstance(unit_tok, str):
            raise ExpressionError(
                f"DATEDIFF first argument must be a string (unit), got {type(unit_tok)}"
            )
        unit = unit_tok.lower()

        self._expect("COMMA")
        start = self._expr()
        self._expect("COMMA")
        end = self._expr()
        self._expect("RPAREN")

        if not isinstance(start, datetime) or not isinstance(end, datetime):
            raise ExpressionError(
                "DATEDIFF requires datetime arguments for start and end."
            )

        delta = end - start
        total_seconds = delta.total_seconds()

        if unit in ("second", "seconds"):
            return int(total_seconds)
        elif unit in ("minute", "minutes"):
            return int(total_seconds / 60)
        elif unit in ("hour", "hours"):
            return int(total_seconds / 3600)
        elif unit in ("day", "days"):
            return delta.days
        else:
            raise ExpressionError(f"Unsupported DATEDIFF unit: {unit}")
