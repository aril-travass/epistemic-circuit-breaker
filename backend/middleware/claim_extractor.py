import ast
import hashlib
import re
from enum import Enum
from typing import Any, Dict, List, Optional, Set
from pydantic import BaseModel, Field


class ClaimType(str, Enum):
    LIBRARY_IMPORT = "LIBRARY_IMPORT"
    FUNCTION_SIGNATURE = "FUNCTION_SIGNATURE"
    RETURN_TYPE = "RETURN_TYPE"
    ALGORITHMIC_ASSUMPTION = "ALGORITHMIC_ASSUMPTION"
    SYSTEM_CONSTRAINT = "SYSTEM_CONSTRAINT"


class AtomicClaim(BaseModel):
    claim_id: str
    claim_type: ClaimType
    statement: str
    confidence: float = 1.0
    source_line: Optional[str] = None
    metadata: Dict[str, Any] = Field(default_factory=dict)


class ClaimExtractionResult(BaseModel):
    claims: List[AtomicClaim]
    raw_text: str
    total_claims: int


STOP_WORDS: Set[str] = {
    "and", "or", "not", "the", "a", "an", "utilize", "use", "apply", "run",
    "to", "all", "in", "is", "for", "with", "from", "as", "by", "on", "of",
    "must", "should", "will", "data", "none", "true", "false", "it", "its"
}


def _make_deterministic_id(claim_type: ClaimType, text: str) -> str:
    """Generates a stable, reproducible claim ID based on type and content hash."""
    clean = text.strip().lower()
    h = hashlib.sha256(f"{claim_type.value}:{clean}".encode("utf-8")).hexdigest()[:12]
    return f"claim_{claim_type.value.lower()}_{h}"


def _clean_token(s: str) -> str:
    """Strips markdown delimiters, punctuation, and whitespace."""
    return s.strip(" \t\n\r`'\",():;*")


def _extract_from_ast_node(node: ast.AST, source_line: Optional[str] = None) -> List[AtomicClaim]:
    """Extracts atomic claims from a Python AST node."""
    claims: List[AtomicClaim] = []

    if isinstance(node, ast.Import):
        for alias in node.names:
            mod_name = alias.name.strip()
            if mod_name and mod_name.lower() not in STOP_WORDS:
                statement = f"Module '{mod_name}' is imported."
                claims.append(AtomicClaim(
                    claim_id=_make_deterministic_id(ClaimType.LIBRARY_IMPORT, statement),
                    claim_type=ClaimType.LIBRARY_IMPORT,
                    statement=statement,
                    source_line=source_line,
                    metadata={"module": mod_name, "alias": alias.asname}
                ))

    elif isinstance(node, ast.ImportFrom):
        mod_name = (node.module or "").strip()
        symbols = [alias.name for alias in node.names]
        if mod_name and mod_name.lower() not in STOP_WORDS:
            statement = f"Module '{mod_name}' is imported with symbols {symbols}."
            claims.append(AtomicClaim(
                claim_id=_make_deterministic_id(ClaimType.LIBRARY_IMPORT, statement),
                claim_type=ClaimType.LIBRARY_IMPORT,
                statement=statement,
                source_line=source_line,
                metadata={"module": mod_name, "symbols": symbols}
            ))

    elif isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
        args = [arg.arg for arg in node.args.args]
        ret_type = ast.unparse(node.returns) if node.returns else None
        fn_stmt = f"Function '{node.name}' is defined with signature ({', '.join(args)})."
        claims.append(AtomicClaim(
            claim_id=_make_deterministic_id(ClaimType.FUNCTION_SIGNATURE, fn_stmt),
            claim_type=ClaimType.FUNCTION_SIGNATURE,
            statement=fn_stmt,
            source_line=source_line,
            metadata={"function_name": node.name, "arguments": args, "return_type": ret_type}
        ))
        if ret_type:
            ret_stmt = f"Function '{node.name}' returns type '{ret_type}'."
            claims.append(AtomicClaim(
                claim_id=_make_deterministic_id(ClaimType.RETURN_TYPE, ret_stmt),
                claim_type=ClaimType.RETURN_TYPE,
                statement=ret_stmt,
                source_line=source_line,
                metadata={"function_name": node.name, "return_type": ret_type}
            ))

    return claims


def extract_atomic_claims(text: str, context_prompt: Optional[str] = None) -> ClaimExtractionResult:
    """
    Extracts atomic semantic claims from text, agent plans, markdown specifications, or code.
    Uses a hybrid approach combining Python AST inspection with robust regex-based extraction.
    """
    claims: List[AtomicClaim] = []
    seen_keys: Set[str] = set()

    def add_claim(c: AtomicClaim) -> None:
        key = f"{c.claim_type.value}:{c.statement.strip().lower()}"
        if key not in seen_keys:
            seen_keys.add(key)
            claims.append(c)

    if not text or not text.strip():
        return ClaimExtractionResult(claims=[], raw_text=text or "", total_claims=0)

    # 1. Structural AST Parsing: extract from markdown fenced code blocks
    code_blocks = re.findall(r"```(?:python)?\s*\n?(.*?)\n?```", text, re.DOTALL | re.IGNORECASE)
    for block in code_blocks:
        try:
            tree = ast.parse(block)
            for node in ast.walk(tree):
                for c in _extract_from_ast_node(node):
                    add_claim(c)
        except SyntaxError:
            pass

    # 2. Structural AST Parsing: attempt to parse entire text as pure Python if no code blocks were found
    if not code_blocks:
        try:
            tree = ast.parse(text)
            for node in ast.walk(tree):
                for c in _extract_from_ast_node(node):
                    add_claim(c)
        except SyntaxError:
            pass

    # 3. Regex and Semantic Pattern Parsing line by line
    lines = text.splitlines()
    for line in lines:
        stripped = line.strip()
        if not stripped:
            continue

        # Ignore markdown headers that don't contain directives or claims
        if re.match(r"^#{1,6}\s+[\w\s\.:-]+$", stripped) and not any(
            kw in stripped.lower() for kw in ["directive", "must", "import", "critical"]
        ):
            continue

        # A. Pattern: 'from <module> import <symbols>'
        from_matches = list(re.finditer(
            r"(?:^|[\s`\'\"\(])from\s+([a-zA-Z_][a-zA-Z0-9_\.]*)\s+import\s+([a-zA-Z0-9_,\s\*]+)",
            stripped
        ))
        has_from_import = False
        for m in from_matches:
            has_from_import = True
            mod = _clean_token(m.group(1))
            raw_symbols = m.group(2)
            raw_symbols = re.split(r"[`\n#;]", raw_symbols)[0]
            symbols = [_clean_token(s) for s in raw_symbols.split(",") if _clean_token(s)]
            if mod and mod.lower() not in STOP_WORDS:
                stmt = f"Module '{mod}' is imported with symbols {symbols}."
                add_claim(AtomicClaim(
                    claim_id=_make_deterministic_id(ClaimType.LIBRARY_IMPORT, stmt),
                    claim_type=ClaimType.LIBRARY_IMPORT,
                    statement=stmt,
                    source_line=stripped,
                    metadata={"module": mod, "symbols": symbols}
                ))

        # B. Pattern: standalone 'import <module1>, <module2>'
        if not has_from_import:
            import_matches = re.finditer(
                r"(?:^|[\s`\'\"\(])import\s+([a-zA-Z_][a-zA-Z0-9_\.]*(?:\s*,\s*[a-zA-Z_][a-zA-Z0-9_\.]*)*)",
                stripped
            )
            for m in import_matches:
                raw_mods = m.group(1).split(",")
                for raw in raw_mods:
                    mod = _clean_token(raw)
                    if mod and mod.lower() not in STOP_WORDS and not mod.startswith("self."):
                        stmt = f"Module '{mod}' is imported."
                        add_claim(AtomicClaim(
                            claim_id=_make_deterministic_id(ClaimType.LIBRARY_IMPORT, stmt),
                            claim_type=ClaimType.LIBRARY_IMPORT,
                            statement=stmt,
                            source_line=stripped,
                            metadata={"module": mod}
                        ))

        # C. Pattern: Library listing lines, e.g., 'Standard Library: `typing`, `dataclasses`, `time`'
        lib_list_match = re.search(
            r"(?:Standard Library|Dependencies|Imports|Packages|Libraries|Requirements):\s*(.+)$",
            stripped,
            re.IGNORECASE
        )
        if lib_list_match:
            parts = re.split(r"[,;]\s*", lib_list_match.group(1))
            for p in parts:
                cleaned = _clean_token(p)
                if cleaned and re.match(r"^[a-zA-Z_][a-zA-Z0-9_\.]*$", cleaned) and cleaned.lower() not in STOP_WORDS:
                    stmt = f"Module '{cleaned}' is imported."
                    add_claim(AtomicClaim(
                        claim_id=_make_deterministic_id(ClaimType.LIBRARY_IMPORT, stmt),
                        claim_type=ClaimType.LIBRARY_IMPORT,
                        statement=stmt,
                        source_line=stripped,
                        metadata={"module": cleaned}
                    ))

        # D. Pattern: explicit function definitions, e.g. 'def parse_data(buffer: bytes) -> dict:'
        fn_match = re.search(
            r"def\s+([a-zA-Z_][a-zA-Z0-9_]*)\s*\((.*?)\)(?:\s*->\s*([a-zA-Z0-9_\[\],\s\.]+))?:",
            stripped
        )
        if fn_match:
            fn_name = fn_match.group(1)
            raw_args = fn_match.group(2)
            ret_type = fn_match.group(3).strip() if fn_match.group(3) else None
            args = [_clean_token(a) for a in raw_args.split(",") if _clean_token(a)]
            stmt = f"Function '{fn_name}' is defined with signature ({', '.join(args)})."
            add_claim(AtomicClaim(
                claim_id=_make_deterministic_id(ClaimType.FUNCTION_SIGNATURE, stmt),
                claim_type=ClaimType.FUNCTION_SIGNATURE,
                statement=stmt,
                source_line=stripped,
                metadata={"function_name": fn_name, "arguments": args, "return_type": ret_type}
            ))
            if ret_type:
                ret_stmt = f"Function '{fn_name}' returns type '{ret_type}'."
                add_claim(AtomicClaim(
                    claim_id=_make_deterministic_id(ClaimType.RETURN_TYPE, ret_stmt),
                    claim_type=ClaimType.RETURN_TYPE,
                    statement=ret_stmt,
                    source_line=stripped,
                    metadata={"function_name": fn_name, "return_type": ret_type}
                ))

        # E. Pattern: System Constraints
        constraint_match = re.search(
            r"^\s*(?:-\s*)?(?:\[?[A-Z\s_-]*DIRECTIVE\]?|CRITICAL|CONSTRAINT|REQUIREMENT|INVARIANT|MUST(?:\s+NOT)?|ENSURE|ENFORCE|STRICTLY|NEVER)\b[:\s]*(.+)$",
            stripped,
            re.IGNORECASE
        )
        if constraint_match and not fn_match:
            clean_line = re.sub(r"^[-*\d\.\s]+", "", stripped).strip()
            if len(clean_line) > 5:
                stmt = f"System constraint: {clean_line}"
                add_claim(AtomicClaim(
                    claim_id=_make_deterministic_id(ClaimType.SYSTEM_CONSTRAINT, stmt),
                    claim_type=ClaimType.SYSTEM_CONSTRAINT,
                    statement=stmt,
                    source_line=stripped,
                    metadata={"constraint": clean_line}
                ))

        # F. Pattern: Algorithmic Assumptions & Roadmap steps
        algo_match = re.search(
            r"^\s*(?:(?:\d+\.|\*|-)\s+)?(?:Assume|Assumption|Algorithm|Architecture|Implement|Apply|Invoke|Format|Validate|Roadmap|Step)\b[:\s]*(.+)$",
            stripped,
            re.IGNORECASE
        )
        if algo_match and not constraint_match and not fn_match and not lib_list_match:
            clean_line = re.sub(r"^[-*\d\.\s]+", "", stripped).strip()
            if len(clean_line) > 8:
                stmt = f"Algorithmic plan step: {clean_line}"
                add_claim(AtomicClaim(
                    claim_id=_make_deterministic_id(ClaimType.ALGORITHMIC_ASSUMPTION, stmt),
                    claim_type=ClaimType.ALGORITHMIC_ASSUMPTION,
                    statement=stmt,
                    source_line=stripped,
                    metadata={"assumption": clean_line}
                ))

    # Incorporate context prompt constraints if provided
    if context_prompt:
        for line in context_prompt.splitlines():
            line_str = line.strip()
            if line_str.startswith("-") or line_str.startswith("*"):
                c_text = line_str.lstrip("-* ").strip()
                if c_text:
                    stmt = f"System constraint: {c_text}"
                    add_claim(AtomicClaim(
                        claim_id=_make_deterministic_id(ClaimType.SYSTEM_CONSTRAINT, stmt),
                        claim_type=ClaimType.SYSTEM_CONSTRAINT,
                        statement=stmt,
                        source_line=line_str,
                        metadata={"constraint": c_text, "from_context": True}
                    ))

    return ClaimExtractionResult(
        claims=claims,
        raw_text=text,
        total_claims=len(claims)
    )
