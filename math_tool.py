import re
import sympy as sp


class MathTool:
    def __init__(self):
        pass

    def _normalize(self, query: str) -> str:
        """
        Normalize human-readable math into machine-readable format.
        """
        return (
            query.lower()
            .replace("×", "*")
            .replace("x", "*")
            .replace("÷", "/")
            .replace("−", "-")
            .replace("–", "-")
            .replace("—", "-")
            .replace("^", "**")
            .replace("plus", "+")
            .replace("minus", "-")
            .replace("times", "*")
            .replace("multiplied by", "*")
            .replace("multiply", "*")
            .replace("divide", "/")
            .replace("divided by", "/")
        )

    def _extract_expression(self, query: str) -> str:
        """
        Extract only valid mathematical tokens.
        """
        normalized = self._normalize(query)

        # Keep numbers, operators, parentheses, decimal points
        tokens = re.findall(r"\d+(?:\.\d+)?|[\+\-\*\/\(\)]", normalized)

        return "".join(tokens)

    def solve(self, query: str) -> str:
        """
        Main function used by agent_controller.
        """
        try:
            expr_str = self._extract_expression(query)

            if not expr_str:
                return "No valid math expression found."

            expr = sp.sympify(expr_str, evaluate=True)
            result = sp.N(expr)

            # Clean integer output
            if getattr(result, "is_Integer", False):
                return str(int(result))

            try:
                float_val = float(result)
                if float_val.is_integer():
                    return str(int(float_val))
                return str(float_val)
            except Exception:
                return str(result)

        except Exception as e:
            return f"Math Error: {str(e)}"


# ✅ Singleton instance (IMPORTANT for your agent)
math_tool = MathTool()