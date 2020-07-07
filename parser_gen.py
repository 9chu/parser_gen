#!python3
# -*- coding: utf-8 -*-
# parser_gen
#   一个 LR(1)/LALR 语法解析器生成工具。
# Author: chu
# Email: 1871361697@qq.com
# License: MIT License
import json
from typing import List, Set, Dict, Tuple, Optional


# ---------------------------------------- 文法解析器部分 ----------------------------------------
# 文法解析器用于解析文法文件，一个文法文件包含语法的终极符、非终结符和对应的产生式及相关的属性。
#
# 终结符使用下述方式声明：
#   term 标识符 {% 替换 %} ;
# 其中，标识符用于指定终结符的名称，可以由非数字开头的若干数字、字母或者下划线构成（下同），需要注意的是单独的_会被识别为关键词。
# 替换部分应当填写一个C/C++类型，当语法制导翻译遇到一个标识符时可以给出对应的C/C++类型的值供用户代码使用。
# 若替换部分留空，则该标识符的值不可在翻译过程中被使用。
# 此外，为了支撑算符优先冲突解决规则，可以在标识符后面使用关键字 assoc 和 prec 来指定左结合或右结合以及对应的优先级，例如：
#   term minus assoc(left) prec(1) {% Tokenizer::Token %};
# 其中 assoc 可以接 left、right 或者 none，表明左结合、右结合或者无结合性。
# 需要注意的是，在解决冲突时，如果发现算符无结合性则会产生错误，若不指定结合性，则会按照其他规约规则自动解决冲突。
# 其中 prec 用于指定算符优先级，算符优先级高的表达式会在移进-规约冲突中被优先选择。
#
# 非终结符使用下述方式声明：
#   nonterm 标识符 {% 替换 %};
# 具体规则和终结符一致，但是不可以声明结合性或者优先级，其他内容不再赘述。
#
# 声明完终结符和非终结符后可以声明语法规则，举例如下：
#   grammar {
#     BinExp -> Exp(lhs) BinOp(op) Exp(rhs) {% return Ast::BinExp(lhs, rhs, op); %};
#     BinOp -> minus {%
#       return Ast::BinOp::Minus;
#     %};
#     BinOp -> plus {%
#       return Ast::BinOp::Plus;
#     %};
#   }
# 语法规则定义在Grammar块中，一个产生式具备下述形式：
#   非终结符 -> 符号1 ( 标识符1 ) 符号2 ( 标识符2 ) ... {% 替换 %} ;
# 其中，非终结符指示从哪个终结符推导而来，整个产生式在规约后将会具备该终结符对应的类型。
# 符号1..n 指示产生式的构成，每个符号可以接一个标识符，将会在生成代码中使用符号对应的类型捕获值给解析器代码使用。
# 需要注意，首条规则被作为入口规则产生文法。此外如果产生式不规约任何符号，需要使用特殊的语法来声明：
#   非终结符 -> _ {% 替换 %};
# 另外，为了支持单目运算符的特殊优先级，产生式本身可以指定一个独立的优先级，例如：
#   grammar {
#     UnaryExp -> minus Exp(rhs) prec(10) {% ... %};
#   }
# 此时，prec 必须在产生式末尾，当生成器在解决 BinExp 和 UnaryExp 的冲突时会优先匹配 UnaryExp。
#
# 最后，在进行代码生成时，你可以使用 Json 来向生成器传递参数，这些参数会被用于在模板中替换对应的变量：
#   generator {%
#     {
#       "namespace": "Test",
#       "class_name": "MyParser"
#     }
#   %}
#
# 附录：关键词表
#   _ term nonterm grammar generator assoc prec left right none
#
# 附录：规约/移进冲突解决规则：
#   下述规则被依次用于解决规约/移进冲突：
#     1. 尝试使用算符优先和结合性规则进行解决；
#     2. 采取移进规则解决；
#   下述规则被依次用于解决规约/规约冲突：
#     1. 依照生成式的定义顺序解决，先定义的生成式会先被用于解决冲突；
#

class Symbol:
    """
    符号

    标识一个终结符或者非终结符。
    符号不覆盖__eq__和__hash__，因为在一个实例中应该是唯一的。
    """
    def __init__(self, t: int, id: str, replace: Optional[str] = None, assoc: int = 0, prec: int = 0, line: int=0):
        self._type = t
        self._id = id
        self._replace = None if replace is None else replace.strip()
        self._assoc = assoc
        self._prec = prec
        self._line = line

    def __repr__(self):
        return self._id

    def type(self) -> int:
        """
        获取符号类型
        :return: 符号类型
        """
        return self._type

    def id(self) -> str:
        """
        获取标识符
        :return: 标识符
        """
        return self._id

    def replace(self) -> Optional[str]:
        """
        获取替换文本
        :return: 替换文本
        """
        return self._replace

    def associativity(self) -> int:
        """
        获取结合性
        :return: 结合性
        """
        return self._assoc

    def precedence(self) -> int:
        """
        获取优先级
        :return: 优先级
        """
        return self._prec

    def line_defined(self) -> int:
        """
        获取符号在源码中定义的行号
        :return: 行号
        """
        return self._line


SYMBOL_TESTER = -2  # special symbol '#', for generating LALR parser
SYMBOL_ENTRY = -1  # special symbol '@', for extending the grammar
SYMBOL_EOF = 0  # special symbol '$'
SYMBOL_TERMINAL = 1  # terminal symbol
SYMBOL_NON_TERMINAL = 2  # non-terminal symbol

ASSOC_UNDEF = 0
ASSOC_LEFT = 1
ASSOC_RIGHT = 2
ASSOC_NONE = 3

kEofSymbol = Symbol(SYMBOL_EOF, "$", "")
kEntrySymbol = Symbol(SYMBOL_ENTRY, "@", "")
kTesterSymbol = Symbol(SYMBOL_TESTER, "#", "")


class Production:
    """
    产生式

    由一系列符号构成。
    """
    def __init__(self, left: Symbol, right: List[Symbol], binding: Dict[int, str], replace: Optional[str] = None,
                 prec: int = 0, line: int = 0):
        self._left = left
        self._right = right
        self._binding = binding
        self._replace = replace
        self._prec = prec
        self._line = line

    def __repr__(self):
        if self._prec != 0:
            return "%s -> %s prec(%d)" % (repr(self._left), " ".join([repr(x) for x in self._right]), self._prec)
        return "%s -> %s" % (repr(self._left), " ".join([repr(x) for x in self._right]))

    def __len__(self):
        return len(self._right)

    def __getitem__(self, item):
        assert isinstance(item, int)
        return self._right[item]

    def __eq__(self, obj) -> bool:  # binding, replace, prec, line 不参与比较
        if not isinstance(obj, Production):
            return False
        if self._left != obj._left:
            return False
        if len(self._right) != len(obj._right):
            return False
        for i in range(0, len(self._right)):
            if self._right[i] != obj._right[i]:
                return False
        return True

    def __ne__(self, obj) -> bool:
        return not self == obj

    def __hash__(self) -> int:
        ret = hash(self._left)
        for i in range(0, len(self._right)):
            ret = ret ^ hash(self._right[i])
        return ret

    def left(self) -> Symbol:
        """
        获取产生式对应的非终结符。
        :return: 非终结符
        """
        return self._left

    def binding(self) -> Dict[int, str]:
        """
        获取绑定参数名的映射情况。
        :return: 绑定参数映射表
        """
        return self._binding

    def replace(self) -> Optional[str]:
        """
        获取产生式对应的替代文本。
        :return: 替代文本
        """
        return self._replace

    def precedence(self) -> int:
        """
        获取优先级
        :return: 优先级
        """
        return self._prec

    def line_defined(self) -> int:
        """
        获取符号在源码中定义的行号
        :return: 行号
        """
        return self._line


class ParseError(Exception):
    """
    解析错误
    """
    def __init__(self, message: str, line: int, col: Optional[int] = None):
        Exception.__init__(self, message)
        self._message = message
        self._line = line
        self._col = col

    def __str__(self):
        if self._col is not None:
            return f"{self._message} (line {self._line}, col {self._col})"
        return f"{self._message} (line {self._line})"

    def message(self):
        return self._message

    def line(self):
        return self._line

    def col(self):
        return self._col


class SourceReader:
    """
    源代码读取器
    """
    def __init__(self, filename):
        with open(filename, "r", encoding="utf-8") as f:
            self._content = f.read()
        self._pos = 0
        self._line = 1
        self._col = 0

    def pos(self):
        return self._pos

    def line(self):
        return self._line

    def col(self):
        return self._col

    def peek(self):
        if self._pos >= len(self._content):
            return '\0'
        return self._content[self._pos]

    def read(self):
        ch = self.peek()
        if ch == '\0':
            return ch
        self._pos = self._pos + 1
        self._col = self._col + 1
        if ch == '\n':
            self._line = self._line + 1
            self._col = 0
        return ch

    def raise_error(self, msg):
        raise ParseError(msg, self._line, self._col)


TOKEN_EOF = 0
TOKEN_IDENTIFIER = 1  # 标识符
TOKEN_LITERAL = 2  # 替换用文本
TOKEN_INTEGER = 3  # 整数
TOKEN_EOD = 4  # 分号 ;
TOKEN_DEDUCE = 5  # 推导符号 ->
TOKEN_BEGIN_BLOCK = 6  # {
TOKEN_END_BLOCK = 7  # }
TOKEN_BEGIN_ARG = 8  # (
TOKEN_END_ARG = 9  # )
TOKEN_EMPTY = 10  # 关键词 _
TOKEN_TERM = 11  # 关键词 term
TOKEN_NON_TERM = 12  # 关键词 nonterm
TOKEN_GRAMMAR = 13  # 关键词 grammar
TOKEN_GENERATOR = 14  # 关键词 generator
TOKEN_ASSOC = 15  # 关键词 assoc
TOKEN_PREC = 16  # 关键词 prec
TOKEN_LEFT = 17  # 关键词 left
TOKEN_RIGHT = 18  # 关键词 right
TOKEN_NONE = 19  # 关键词 none


class GrammarDocument:
    """
    语法文件

    存储语法文件内容并提供解析功能。
    使用手写的递归下降来实现解析。

    @mq
    - 没有parser gen，要怎么解析语法文件
    - 写parser啊
    - 没有parser gen怎么写parser
    - 那就写parser gen
    - 写parser gen怎么解析语法规则！！
    - 写parser！！！
    """
    def __init__(self):
        self._productions = []  # type: List[Production]
        self._symbols = set()  # type: Set[Symbol]
        self._terminals = set()  # type: Set[Symbol]
        self._non_terminals = set()  # type: Set[Symbol]
        self._generator_args = None  # type: Optional[Dict]

    def clear(self):
        self._productions = []  # type: List[Production]
        self._symbols = set()  # type: Set[Symbol]
        self._terminals = set()  # type: Set[Symbol]
        self._non_terminals = set()  # type: Set[Symbol]
        self._generator_args = None  # type: Optional[Dict]

    def productions(self) -> List[Production]:
        """
        获取所有产生式
        :return: 产生式列表
        """
        return self._productions

    def symbols(self) -> Set[Symbol]:
        """
        获取所有符号
        :return: 符号集合
        """
        return self._symbols

    def terminals(self) -> Set[Symbol]:
        """
        获取终结符号
        :return: 终结符号集合
        """
        return self._terminals

    def non_terminals(self) -> Set[Symbol]:
        """
        获取非终结符号
        :return: 非终结符号集合
        """
        return self._non_terminals

    def generator_args(self) -> Optional[Dict]:
        """
        获取生成器参数
        :return: 参数
        """
        return self._generator_args

    @staticmethod
    def _advance(reader: SourceReader):
        while True:
            if reader.peek() == '\0':
                return TOKEN_EOF, None, reader.line()

            # 跳过空白
            if reader.peek().isspace():
                while reader.peek().isspace():
                    reader.read()
                continue

            # 跳过注释
            if reader.peek() == '/':
                reader.read()
                if reader.peek() != '/':  # 当前语法只有'//'的可能
                    reader.raise_error(f"'/' expected, but found {repr(reader.peek())}")
                reader.read()
                while reader.peek() != '\0' and reader.peek() != '\n':  # 读到末尾
                    reader.read()
                continue

            # 符号
            if reader.peek() == ';':
                line = reader.line()
                reader.read()
                return TOKEN_EOD, None, line
            elif reader.peek() == '-':
                line = reader.line()
                reader.read()
                if reader.peek() != '>':  # 当前语法只有'->'可能
                    reader.raise_error(f"'>' expected, but found {repr(reader.peek())}")
                reader.read()
                return TOKEN_DEDUCE, None, line
            elif reader.peek() == '{':
                line = reader.line()
                reader.read()
                if reader.peek() == '%':
                    reader.read()
                    content = []
                    while True:
                        if reader.peek() == '%':
                            reader.read()
                            if reader.peek() == '}':
                                reader.read()
                                break
                            elif reader.peek() == '%':
                                reader.read()
                                content.append('%')
                            else:
                                reader.raise_error(f"'%' or '}}' expected, but found {repr(reader.peek())}")
                        elif reader.peek() == '\0':
                            reader.raise_error("Unexpected eof")
                        else:
                            content.append(reader.read())
                    return TOKEN_LITERAL, "".join(content), line
                else:
                    return TOKEN_BEGIN_BLOCK, None, line
            elif reader.peek() == '}':
                line = reader.line()
                reader.read()
                return TOKEN_END_BLOCK, None, line
            elif reader.peek() == '(':
                line = reader.line()
                reader.read()
                return TOKEN_BEGIN_ARG, None, line
            elif reader.peek() == ')':
                line = reader.line()
                reader.read()
                return TOKEN_END_ARG, None, line

            # 关键词/Identifier/数字
            content = []
            if reader.peek().isalpha() or reader.peek() == '_':
                line = reader.line()
                while reader.peek().isalnum() or reader.peek() == '_':
                    content.append(reader.read())
                identifier = "".join(content)
                if identifier == "_":
                    return TOKEN_EMPTY, identifier, line
                elif identifier == "term":
                    return TOKEN_TERM, identifier, line
                elif identifier == "nonterm":
                    return TOKEN_NON_TERM, identifier, line
                elif identifier == "grammar":
                    return TOKEN_GRAMMAR, identifier, line
                elif identifier == "generator":
                    return TOKEN_GENERATOR, identifier, line
                elif identifier == "assoc":
                    return TOKEN_ASSOC, identifier, line
                elif identifier == "prec":
                    return TOKEN_PREC, identifier, line
                elif identifier == "left":
                    return TOKEN_LEFT, identifier, line
                elif identifier == "right":
                    return TOKEN_RIGHT, identifier, line
                elif identifier == "none":
                    return TOKEN_NONE, identifier, line
                return TOKEN_IDENTIFIER, identifier, line
            if reader.peek().isnumeric():
                line = reader.line()
                while reader.peek().isnumeric():
                    content.append(reader.read())
                return TOKEN_INTEGER, int("".join(content)), line
            reader.raise_error(f"Unexpected character '{repr(reader.peek())}'")

    def parse(self, filename):
        reader = SourceReader(filename)
        symbols = {}
        productions = []
        production_set = set()
        generator_args = None
        while True:
            token, value, line = GrammarDocument._advance(reader)
            if token == TOKEN_EOF:
                break
            elif token == TOKEN_TERM:
                # read identifier
                token, identifier, line = GrammarDocument._advance(reader)
                if token != TOKEN_IDENTIFIER:
                    raise ParseError("Identifier required parsing term statement", line)
                if identifier in symbols:
                    raise ParseError(f"Terminated symbol \"{identifier}\" redefined", line)
                replace = None
                def_line = line
                # read assoc or prec
                assoc = None
                prec = None
                while True:
                    token, value, line = GrammarDocument._advance(reader)
                    if token == TOKEN_ASSOC:
                        if assoc is not None:
                            raise ParseError("Associate type redefined", line)
                        token, _, line = GrammarDocument._advance(reader)
                        if token != TOKEN_BEGIN_ARG:
                            raise ParseError("'(' expected parsing associate type", line)
                        token, _, line = GrammarDocument._advance(reader)
                        if token == TOKEN_LEFT:
                            assoc = ASSOC_LEFT
                        elif token == TOKEN_RIGHT:
                            assoc = ASSOC_RIGHT
                        elif token == TOKEN_NONE:
                            assoc = ASSOC_NONE
                        else:
                            raise ParseError("'left', 'right' or 'none' expected parsing associate type", line)
                        token, _, line = GrammarDocument._advance(reader)
                        if token != TOKEN_END_ARG:
                            raise ParseError("')' expected parsing associate type", line)
                    elif token == TOKEN_PREC:
                        if prec is not None:
                            raise ParseError("Precedence redefined", line)
                        token, _, line = GrammarDocument._advance(reader)
                        if token != TOKEN_BEGIN_ARG:
                            raise ParseError("'(' expected parsing precedence", line)
                        token, prec, line = GrammarDocument._advance(reader)
                        if token != TOKEN_INTEGER:
                            raise ParseError("Integer expected parsing precedence", line)
                        token, _, line = GrammarDocument._advance(reader)
                        if token != TOKEN_END_ARG:
                            raise ParseError("')' expected parsing associate type", line)
                    else:
                        break
                # replace
                if token == TOKEN_LITERAL:
                    replace = value
                    token, _, line = GrammarDocument._advance(reader)
                if token != TOKEN_EOD:
                    raise ParseError("End of definition required", line)
                symbols[identifier] = Symbol(SYMBOL_TERMINAL, identifier, replace,
                                             ASSOC_UNDEF if assoc is None else assoc,
                                             0 if prec is None else prec,
                                             def_line)
            elif token == TOKEN_NON_TERM:
                # read identifier
                token, identifier, line = GrammarDocument._advance(reader)
                if token != TOKEN_IDENTIFIER:
                    raise ParseError("Identifier required parsing term statement", line)
                if identifier in symbols:
                    raise ParseError(f"Non-terminated symbol \"{identifier}\" redefined", line)
                replace = None
                def_line = line
                # replace
                token, value, line = GrammarDocument._advance(reader)
                if token == TOKEN_LITERAL:
                    replace = value
                    token, _, line = GrammarDocument._advance(reader)
                if token != TOKEN_EOD:
                    raise ParseError("End of definition required", line)
                symbols[identifier] = Symbol(SYMBOL_NON_TERMINAL, identifier, replace, ASSOC_UNDEF, 0, def_line)
            elif token == TOKEN_GRAMMAR:
                token, _, line = GrammarDocument._advance(reader)
                if token != TOKEN_BEGIN_BLOCK:
                    raise ParseError("'{' required parsing grammar block", line)
                while True:
                    token, identifier, line = GrammarDocument._advance(reader)
                    if token == TOKEN_END_BLOCK:  # }
                        break
                    elif token != TOKEN_IDENTIFIER:
                        raise ParseError("Identifier required parsing production expression", line)

                    # identifier
                    if identifier not in symbols:
                        raise ParseError(f"Undefined symbol \"{identifier}\" parsing production expression", line)
                    # ->
                    token, _, line = GrammarDocument._advance(reader)
                    if token != TOKEN_DEDUCE:
                        raise ParseError("Deduce operator required parsing production expression", line)
                    right = []
                    replace = None
                    prec = None
                    empty_production = False
                    def_line = line
                    binding = {}
                    while True:
                        token, value, line = GrammarDocument._advance(reader)
                        if token == TOKEN_EOD:  # ;
                            if not empty_production and len(right) == 0:
                                raise ParseError("Symbol expected but found ';' parsing production expression", line)
                            break
                        elif token == TOKEN_LITERAL:
                            if not empty_production and len(right) == 0:
                                raise ParseError("Symbol expected but found replacement literal", line)
                            replace = value
                            token, _, line = GrammarDocument._advance(reader)
                            if token != TOKEN_EOD:
                                raise ParseError("End of definition required parsing production expression", line)
                            break
                        elif token == TOKEN_EMPTY:
                            if len(right) != 0 or (prec is not None):
                                raise ParseError("Epsilon symbol cannot be placed here parsing production expression",
                                                 line)
                            empty_production = True
                        elif token == TOKEN_PREC:
                            token, _, line = GrammarDocument._advance(reader)
                            if token != TOKEN_BEGIN_ARG:
                                raise ParseError("'(' required parsing precedence", line)
                            token, prec, line = GrammarDocument._advance(reader)
                            if token != TOKEN_INTEGER:
                                raise ParseError("Integer expected parsing precedence", line)
                            token, _, line = GrammarDocument._advance(reader)
                            if token != TOKEN_END_ARG:
                                raise ParseError("')' required parsing precedence", line)
                        elif token == TOKEN_IDENTIFIER:
                            if empty_production or (prec is not None):
                                raise ParseError("Identifier cannot be placed here", line)
                            if value not in symbols:
                                raise ParseError(f"Undefined symbol \"{value}\"", line)
                            right.append(symbols[value])
                        elif token == TOKEN_BEGIN_ARG:
                            if len(right) == 0:
                                raise ParseError("Symbol required for binding argument name", line)
                            if right[len(right) - 1].replace() is None:
                                raise ParseError("Symbol don't have type for binding", line)
                            token, arg_id, line = GrammarDocument._advance(reader)
                            if token != TOKEN_IDENTIFIER:
                                raise ParseError("Identifier required parsing binding argument", line)
                            token, _, line = GrammarDocument._advance(reader)
                            if token != TOKEN_END_ARG:
                                raise ParseError("')' expected parsing binding argument", line)
                            binding[len(right) - 1] = arg_id
                        else:
                            raise ParseError("Unexpected token", line)
                    assert len(right) > 0 or empty_production
                    # calc prec if user not defined
                    if prec is None:
                        for e in reversed(right):
                            if e.type() == SYMBOL_TERMINAL:
                                prec = e.precedence()
                        if prec is None:
                            prec = 0
                    production = Production(symbols[identifier], right, binding, replace, prec, def_line)
                    if production in production_set:
                        raise ParseError(f"Production \"{production}\" redefined", def_line)
                    if (production.left().replace() is not None) and (production.replace() is None):
                        raise ParseError(f"Action body expected for production \"{production}\"", def_line)
                    productions.append(production)
                    production_set.add(production)
                token, _, line = GrammarDocument._advance(reader)
                if token != TOKEN_EOD:
                    raise ParseError("End of definition required parsing grammar block", line)
            elif token == TOKEN_GENERATOR:
                if generator_args is not None:
                    raise ParseError("Generator arguments is redefined", line)
                try:
                    token, json_args, line = GrammarDocument._advance(reader)
                except Exception as ex:
                    raise ParseError(f"Parsing json error parsing generator block: {ex}", line)
                if token != TOKEN_LITERAL:
                    raise ParseError("String literal required parsing generator block", line)
                token, _, line = GrammarDocument._advance(reader)
                if token != TOKEN_EOD:
                    raise ParseError("';' expected parsing generator block", line)
                generator_args = json.loads(json_args)
            else:
                raise ParseError("Unexpected token", line)
        self._productions = productions
        self._symbols = set([symbols[s] for s in symbols])
        self._terminals = set([s for s in self._symbols if s.type() == SYMBOL_TERMINAL])
        self._non_terminals = set([s for s in self._symbols if s.type() == SYMBOL_NON_TERMINAL])
        self._generator_args = generator_args

# ---------------------------------------- LR(1)/LALR分析器部分 ----------------------------------------
# LR(1)/LALR分析器用于解算状态转移矩阵。
# 通过对文法进行LR分析，可以得到类似下图的转换矩阵：
#      x opt eq $  | S E  V
#   0 s2 s4        |
#   1           a  |
#   2        r3 r3 |
#   3 s2 s4        |   g8 g7
# ……下略
# 其中，表头表示向前看符号，每一行代表一个解析器状态，每一个格表明在看到下一个输入符号时需要进行的动作：
#   sX 表明一个移进操作，在移入下一个符号后跳转到状态X
#   rX 表明一个规约操作，在看到当前符号时按照产生式X进行规约，弹出解析栈顶部的|X|个元素
#   gX 表明在规约操作后，在看到栈顶符号为这个格子对应的符号时，转移状态到X状态
# 同时分析器会依据之前的规则对 SR冲突、RR冲突 进行解决


class ExtendProduction:
    """
    扩展生成式（项）

    增加当前位置和向前看符号来计算闭包。
    """
    def __init__(self, raw: Production, index: int, lookahead: Set[Symbol]):
        assert len(raw) >= index
        self._production = raw
        self._index = index
        self._lookahead = lookahead

    def __repr__(self):
        right = [repr(x) for x in self._production]
        right.insert(self._index, "·")
        return "(%s -> %s, %s)" % (repr(self._production.left()), " ".join(right), self._lookahead)

    def __len__(self):
        return len(self._production)

    def __getitem__(self, item):
        assert isinstance(item, int)
        return self._production[item]

    def __eq__(self, obj) -> bool:
        if not isinstance(obj, ExtendProduction):
            return False
        if self._index != obj._index:
            return False
        if self._production != obj._production:
            return False
        if self._lookahead != obj._lookahead:
            return False
        return True

    def __ne__(self, obj) -> bool:
        return not self == obj

    def __hash__(self) -> int:
        ret = hash(self._index)
        for x in self._lookahead:
            ret = ret ^ hash(x)
        ret = ret ^ hash(self._production)
        return ret

    def production(self) -> Production:
        """
        获取原始产生式
        :return: 产生式
        """
        return self._production

    def index(self) -> int:
        """
        获取当前分析位置
        :return: 位置
        """
        return self._index

    def lookahead(self) -> Set[Symbol]:
        """
        关联的向前看符号
        :return: 符号
        """
        return self._lookahead


class ExtendProductionSet:
    """
    扩展生成式集合（项集）
    """
    def __init__(self, s: Set[ExtendProduction], state_id: Optional[int]):
        self._set = s
        self._state = state_id

    def __repr__(self):
        return repr(self._set)

    def __len__(self):
        return len(self._set)

    def __eq__(self, obj) -> bool:
        if not isinstance(obj, ExtendProductionSet):
            return False
        return self._set == obj._set

    def __ne__(self, obj) -> bool:
        return not self == obj

    def __hash__(self) -> int:
        ret = 0
        for x in self._set:
            ret = ret ^ hash(x)
        return ret

    def __iter__(self):
        return iter(self._set)

    def __repr__(self):
        ret = ["["]
        for e in self._set:
            ret.append(f"  {repr(e)}")
        ret.append("]")
        return "\n".join(ret)

    def state(self):
        """
        获取状态ID
        :return: 状态ID
        """
        return self._state

    def add(self, x: ExtendProduction):
        """
        向集合添加产生式
        :param x: 产生式
        """
        self._set.add(x)

    def union(self, x):
        """
        合并其他集合
        :param x: 集合
        """
        if isinstance(x, set):
            self._set = self._set.union(x)
        else:
            assert isinstance(x, ExtendProductionSet)
            self._set = self._set.union(x._set)

    def clone(self):
        """
        产生副本
        :return: 项集
        """
        return ExtendProductionSet(set(self._set), self._state)


ACTION_ACCEPT = 1
ACTION_GOTO = 2  # Shift和Goto可以用一个动作表示，因为对于非终结符总是Goto的，对于终结符总是Shift的
ACTION_REDUCE = 3  # 规约动作


class Action:
    """
    语法动作
    """
    def __init__(self, action: int, arg, ref_state: ExtendProductionSet, ref_prod: ExtendProduction):
        self._action = action
        self._arg = arg
        self._ref_state = ref_state
        self._ref_prod = ref_prod

    def action(self) -> int:
        """
        获取动作
        :return: 动作
        """
        return self._action

    def arg(self):
        """
        获取动作参数
        :return: 参数
        """
        return self._arg

    def ref_state(self) -> ExtendProductionSet:
        """
        获取关联的状态集合
        :return: 项集
        """
        return self._ref_state

    def ref_prod(self) -> ExtendProduction:
        """
        获取关联的生成式
        :return: 项
        """
        return self._ref_prod


# GRAMMAR_MODE_LR1 = 0
# GRAMMAR_MODE_LALR = 1


class GrammarGenerator:
    def __init__(self, document: GrammarDocument):
        self._doc = document

        # 初始化 NullableSet、FirstSet 和 FollowSet 并计算
        # 注意这个 Set 会包含 kEofSymbol
        self._nullable_set = {}  # type: Dict[Symbol, bool]
        self._first_set = {}  # type: Dict[Symbol, Set[Symbol]]
        self._follow_set = {}  # type: Dict[Symbol, Set[Symbol]]
        self._analyze_nullable_first_follow_set()

        # 初始化扩展符号表
        self._extend_symbols = set(self._doc.symbols())  # type: Set[Symbol]
        self._extend_symbols.add(kEofSymbol)

        # 初始化分析动作表
        self._actions = {}  # type: Dict[Symbol, Dict[int, Tuple[int, Optional[int]]]]
        self._max_state = 0  # 最大的状态ID
        self._reset_actions()

    def _analyze_nullable_first_follow_set(self):
        # 对所有产生式执行拓扑排序的计算，并按照出度从小到大排序
        toposort_states = {}  # type: Dict[Symbol, Dict]
        toposort_results = []  # type: List[Production]

        # 初始化数据集
        for s in self._doc.non_terminals():
            toposort_states[s] = {
                "out": 0,  # 出度
                "from": [],  # 入度
                "visited": False,  # 是否已处理
                "productions": [],  # 从当前非终结符号导出的产生式
            }
        for p in self._doc.productions():
            toposort_states[p.left()]["productions"].append(p)
            for i in range(0, len(p)):
                if p[i].type() == SYMBOL_NON_TERMINAL:
                    toposort_states[p.left()]["out"] += 1
                    toposort_states[p[i]]["from"].append(toposort_states[p.left()])

        # 迭代进行拓扑排序直到集合为空
        while len(toposort_results) < len(self._doc.productions()):
            refs_min = None
            for k in toposort_states:  # 寻找一个出度最小节点
                state = toposort_states[k]
                if state["visited"]:
                    continue
                if refs_min is None or state["out"] < refs_min["out"]:
                    refs_min = state
            assert refs_min is not None
            toposort_results.extend(refs_min["productions"])  # 将当前节点的产生式放入队列
            # 从集合中隐藏当前节点
            refs_min["visited"] = True
            for e in refs_min["from"]:
                e["out"] -= 1
                assert e["out"] >= 0
        assert len(toposort_results) == len(self._doc.productions())

        # 初始化集合
        nullable_set = {kEofSymbol: False}  # type: Dict[Symbol, bool]
        first_set = {kEofSymbol: {kEofSymbol}}  # type: Dict[Symbol, Set[Symbol]]
        follow_set = {kEofSymbol: set()}  # type: Dict[Symbol, Set[Symbol]]
        for s in self._doc.symbols():
            nullable_set[s] = False
            first_set[s] = {s} if s.type() == SYMBOL_TERMINAL else set()
            follow_set[s] = set()

        # 迭代到不动点计算NULLABLE、FIRST集合和FOLLOW集合
        while True:
            stopped = True
            for p in toposort_results:
                s = p.left()

                # 检查产生式是否可空，即产生式中所有项都可空能推导出当前的非终结符可空
                if not nullable_set[s]:  # 对于已经认为可空的永远不会变为非可空
                    nullable = True
                    for i in range(0, len(p)):
                        if not nullable_set[p[i]]:  # 非空
                            nullable = False
                            break
                    if nullable_set[s] != nullable:
                        nullable_set[s] = nullable
                        stopped = False

                # 计算FIRST集
                first = set(first_set[s])
                for i in range(0, len(p)):
                    # 若 p[0..i] 都可空，那么 first[s] = first[s] ∪ first[p[i]]
                    prefix_nullable = True
                    for j in range(0, i):
                        if not nullable_set[p[j]]:
                            prefix_nullable = False
                            break
                    if prefix_nullable:
                        first = first.union(first_set[p[i]])
                    else:
                        break  # 如果中间出现过不可空的，则无需继续看
                if first_set[s] != first:
                    first_set[s] = first
                    stopped = False

                # 计算FOLLOW集
                for i in range(0, len(p)):
                    x = p[i]  # 注意此时计算的目标是产生式中的每个项
                    follow = set(follow_set[x])  # copy
                    # 若 p[i+1..len(p)] 都可空，那么 follow[x] = follow[x] ∪ follow[s]
                    postfix_nullable = True
                    for j in range(i + 1, len(p)):
                        if not nullable_set[p[j]]:
                            postfix_nullable = False
                            break
                    if postfix_nullable:
                        follow = follow.union(follow_set[s])
                    # 若 p[i+1..j] 都可空，那么 follow[x] = follow[x] ∪ first[j]
                    for j in range(i + 1, len(p)):
                        inner_nullable = True
                        for k in range(i + 1, j):
                            if not nullable_set[p[k]]:
                                inner_nullable = False
                                break
                        if inner_nullable:
                            follow = follow.union(first_set[p[j]])
                    if follow_set[x] != follow:
                        follow_set[x] = follow
                        stopped = False
            if stopped:
                break
        self._nullable_set = nullable_set
        self._first_set = first_set
        self._follow_set = follow_set

    def _reset_actions(self):
        for s in self._extend_symbols:
            self._actions[s] = {}
        self._max_state = 0

    def _resolve_conflict(self) -> bool:
        pass  # TODO

    def _populate_action(self, s: Symbol, state: int, act: Tuple[int, Optional[int]]):
        if state in self._actions[s]:
            if self._actions[s][state][0] == ACTION_GOTO:
                left = "shift"
            elif self._actions[s][state][0] == ACTION_REDUCE:
                left = "reduce"
            elif self._actions[s][state][0] == ACTION_ACCEPT:
                left = "accept"
            else:
                assert False
            if act[0] == ACTION_GOTO:
                right = "shift"
            elif act[0] == ACTION_REDUCE:
                right = "reduce"
            elif act[0] == ACTION_ACCEPT:
                right = "accept"
            else:
                assert False
            raise RuntimeError(f"{left}/{right} conflict, symbol {s}, state {state}, action {act}")
        self._actions[s][state] = act

    def _format_actions(self):
        ret = []
        header = self._actions.keys()
        ret.append("|" + "|".join([x.id() for x in header]))
        for state in range(0, self._max_state + 1):
            line = [str(state)]
            for k in header:
                if state in self._actions[k]:
                    action = self._actions[k][state][0]
                    arg = self._actions[k][state][1]
                    if action == ACTION_REDUCE:
                        line.append(f"r{arg}")
                    elif action == ACTION_ACCEPT:
                        line.append("a")
                    elif action == ACTION_GOTO:
                        if k.type() == SYMBOL_NON_TERMINAL:
                            line.append(f"g{arg}")
                        else:
                            line.append(f"s{arg}")
                else:
                    line.append("")
            ret.append("|".join(line))
        return "\n".join(ret)

    def _closure(self, org: ExtendProductionSet):
        ret = org.clone()  # copy
        add = set()
        while True:
            for e in ret:
                if e.index() >= len(e.production()):
                    continue

                x = e.production()[e.index()]
                if x.type() == SYMBOL_TERMINAL:
                    continue
                if x.type() == SYMBOL_EOF:
                    assert (len(e.lookahead()) == 0)
                    continue
                assert(x.type() != SYMBOL_ENTRY)

                # 计算FIRST集
                first = set()
                for i in range(e.index() + 1, len(e.production()) + 1):
                    # 若 p[cur+1..i] 都可空，那么 first[X] = first[X] ∪ first[p[i]]
                    prefix_nullable = True
                    for j in range(e.index() + 1, i):
                        if not self._nullable_set[e.production()[j]]:
                            prefix_nullable = False
                            break
                    if prefix_nullable:
                        if i == len(e.production()):
                            first = first.union(e.lookahead())
                        else:
                            first = first.union(self._first_set[e.production()[i]])
                    else:
                        break  # 如果中间出现过不可空的，则无需继续看

                # 展开终结符
                for p in self._doc.productions():
                    if p.left() == x:
                        for w in first:
                            item = ExtendProduction(p, 0, {w})
                            if item not in ret and item not in add:
                                add.add(item)

            if len(add) == 0:
                break
            ret.union(add)
            add.clear()
        return ret

    def _goto(self, org: ExtendProductionSet, x: Symbol):
        ret = set()
        for e in org:
            if e.index() >= len(e.production()):
                continue
            s = e.production()[e.index()]
            if s != x:
                continue
            p = ExtendProduction(e.production(), e.index() + 1, e.lookahead())
            if p not in ret:
                ret.add(p)
        return self._closure(ExtendProductionSet(ret))

    def process(self):
        # 以首个规则作为入口
        entry_rule = ExtendProduction(Production(kEntrySymbol, [self._doc.productions()[0].left(), kEofSymbol], None),
                                      0, set())
        entry_rule_set = ExtendProductionSet({entry_rule})
        entry_rule_closure = self._closure(entry_rule_set)

        # 初始化状态
        next_state = 1
        states = {entry_rule_closure: 0}  # type: Dict[ExtendProductionSet, int]
        q = [entry_rule_closure]  # type: List[ExtendProductionSet]

        # 计算动作表
        self._reset_actions()
        while len(q) > 0:
            current = q.pop(0)
            state = states[current]

            # 填写规约动作
            for p in current:
                if p.index() >= len(p.production()):
                    p_index = self._doc.productions().index(p.production())
                    for x in p.lookahead():
                        self._populate_action(x, state, (ACTION_REDUCE, p_index))

            # 计算Shift/Goto/Accept
            for x in self._extend_symbols:
                goto = self._goto(current, x)
                if len(goto) == 0:
                    continue
                if x == kEofSymbol:
                    self._populate_action(x, state, (ACTION_ACCEPT, None))
                else:
                    if goto in states:
                        goto_state = states[goto]
                    else:
                        goto_state = next_state
                        next_state += 1
                        states[goto] = goto_state
                        q.append(goto)
                    self._populate_action(x, state, (ACTION_GOTO, goto_state))
        self._max_state = next_state - 1
        print(self._format_actions())


if __name__ == "__main__":
    doc = GrammarDocument()
    doc.parse("sample.txt")
    print(doc.productions())
    print(doc.terminals())
    print(doc.non_terminals())
    analyzer = GrammarGenerator(doc)
    analyzer.process()
