#!python3
# -*- coding: utf-8 -*-
# parser_gen
#   一个 LR(1)/LALR 语法解析器生成工具。
# Author: chu
# Email: 1871361697@qq.com
# License: MIT License
import os
import sys
import json
import argparse
import datetime
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
                 prec: int = 0, line: int = -1, index: int = -1):
        self._left = left
        self._right = right
        self._binding = binding
        self._replace = replace
        self._prec = prec
        self._line = line
        self._index = index

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
        获取产生式对应的非终结符
        :return: 非终结符
        """
        return self._left

    def binding(self) -> Dict[int, str]:
        """
        获取绑定参数名的映射情况
        :return: 绑定参数映射表
        """
        return self._binding

    def replace(self) -> Optional[str]:
        """
        获取产生式对应的替代文本
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

    def index(self) -> int:
        """
        获取产生式在源码中的索引
        :return: 索引
        """
        return self._index


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
                        if prec == 0:
                            raise ParseError("Precedence must large than zero", line)
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
                if (assoc is not None) and (prec is None):
                    raise ParseError("Precedence must be defined while associativity defined", def_line)
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
                            if prec == 0:
                                raise ParseError("Precedence must large than zero", line)
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
                    production = Production(symbols[identifier], right, binding, replace, prec, def_line,
                                            len(productions))
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
    def __init__(self, raw: Production, pos: int, lookahead: Set[Symbol]):
        assert len(raw) >= pos
        self._production = raw
        self._pos = pos
        self._lookahead = lookahead

    def __repr__(self):
        right = [repr(x) for x in self._production]
        right.insert(self._pos, "·")
        return "(%s -> %s, %s)" % (repr(self._production.left()), " ".join(right), self._lookahead)

    def __len__(self):
        return len(self._production)

    def __getitem__(self, item):
        assert isinstance(item, int)
        return self._production[item]

    def __eq__(self, obj) -> bool:
        if not isinstance(obj, ExtendProduction):
            return False
        if self._pos != obj._pos:
            return False
        if self._production != obj._production:
            return False
        if self._lookahead != obj._lookahead:
            return False
        return True

    def __ne__(self, obj) -> bool:
        return not self == obj

    def __hash__(self) -> int:
        ret = hash(self._pos)
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

    def pos(self) -> int:
        """
        获取当前分析位置
        :return: 位置
        """
        return self._pos

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

    def __len__(self):
        return len(self._set)

    def __eq__(self, obj) -> bool:  # state_id不参与比较
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

    def set_state(self, state):
        """
        设置状态ID
        """
        self._state = state

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
    def __init__(self, action: int, arg, ref_state: ExtendProductionSet, ref_symbol: Symbol,
                 ref_prod: Optional[ExtendProduction]):
        self._action = action
        self._arg = arg
        self._ref_state = ref_state
        self._ref_symbol = ref_symbol
        self._ref_prod = ref_prod

        # 参数检查
        if action == ACTION_GOTO:
            assert isinstance(arg, ExtendProductionSet)
        elif action == ACTION_REDUCE:
            assert isinstance(arg, Production)
            assert arg.index() >= 0

    def __repr__(self):
        if self._action == ACTION_ACCEPT:
            return "a"
        elif self._action == ACTION_GOTO:
            assert isinstance(self._arg, ExtendProductionSet)
            if self._ref_symbol.type() == SYMBOL_NON_TERMINAL:
                return f"g{self._arg.state()}"
            else:
                return f"s{self._arg.state()}"
        elif self._action == ACTION_REDUCE:
            assert isinstance(self._arg, Production)
            return f"r{self._arg.index()}"
        return ""

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

    def ref_symbol(self) -> Symbol:
        """
        获取关联的符号
        :return: 符号
        """
        return self._ref_symbol

    def ref_prod(self) -> Optional[ExtendProduction]:
        """
        获取关联的生成式

        对于Shift操作不存在关联的生成式。
        :return: 项
        """
        return self._ref_prod


class GrammarError(Exception):
    """
    解析错误
    """
    def __init__(self, message: str):
        Exception.__init__(self, message)
        self._message = message

    def message(self):
        return self._message


GRAMMAR_MODE_LR1 = 0
GRAMMAR_MODE_LALR = 1


class GrammarAnalyzer:
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
        self._actions = {}  # type: Dict[Symbol, Dict[int, Action]]
        self._max_state = 0  # 最大的状态ID
        self._resolve_rr_conflict = 0  # 解决Reduce/Reduce冲突的次数
        self._resolve_sr_conflict_by_prec = 0  # 解决Reduce/Shift冲突的次数（通过算符优先）
        self._resolve_sr_conflict_by_shift = 0  # 解决Reduce/Shift冲突的次数（通过Shift优先）
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
        self._resolve_rr_conflict = 0
        self._resolve_sr_conflict_by_prec = 0
        self._resolve_sr_conflict_by_shift = 0

    def _closure(self, org: ExtendProductionSet) -> ExtendProductionSet:
        """
        求项集的闭包
        :param org: 原始项集
        :return: 项集的闭包
        """
        ret = org.clone()  # copy
        ret.set_state(-1)  # 需要外部重新赋予状态ID
        add = set()
        while True:
            for e in ret:
                if e.pos() >= len(e.production()):
                    continue

                x = e.production()[e.pos()]
                if x.type() == SYMBOL_TERMINAL:
                    continue
                if x.type() == SYMBOL_EOF:
                    assert (len(e.lookahead()) == 0)
                    continue
                assert(x.type() != SYMBOL_ENTRY)

                # 计算 FIRST 集
                first = set()
                for i in range(e.pos() + 1, len(e.production()) + 1):
                    # 若 p[cur+1..i] 都可空，那么 first[X] = first[X] ∪ first[p[i]]
                    prefix_nullable = True
                    for j in range(e.pos() + 1, i):
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

    def _goto(self, org: ExtendProductionSet, x: Symbol) -> ExtendProductionSet:
        """
        求项集在符号 X 下可以转移到的状态
        :param org: 原始项集
        :param x: 转移符号
        :return: 输出状态
        """
        ret = set()
        for e in org:
            if e.pos() >= len(e.production()):
                continue
            s = e.production()[e.pos()]
            if s != x:
                continue
            p = ExtendProduction(e.production(), e.pos() + 1, e.lookahead())
            if p not in ret:
                ret.add(p)
        return self._closure(ExtendProductionSet(ret, -1))  # 需要外部重新赋予状态ID

    def _populate_action(self, s: Symbol, state: int, act: Action):
        if state in self._actions[s]:  # 冲突解决
            raise_error = True
            conflict_type = 0  # 0: unknown 1: shift/shift冲突 2:shift/reduce冲突 3:reduce/reduce冲突
            conflict_args = ()
            org_action = self._actions[s][state]
            assert state == org_action.ref_state().state()

            # 如果存在Shift/Shift冲突，则抛出错误
            if org_action.action() == ACTION_GOTO and act.action() == ACTION_GOTO:
                assert isinstance(org_action.arg(), ExtendProductionSet)
                assert isinstance(act.arg(), ExtendProductionSet)
                conflict_type = 1
                conflict_args = (s, org_action.ref_state().state(), org_action.arg(), act.arg())

            # 针对Reduce/Reduce的情况，选择优先出现的规则
            if org_action.action() == ACTION_REDUCE and act.action() == ACTION_REDUCE:
                assert isinstance(org_action.arg(), Production)
                assert isinstance(act.arg(), Production)
                assert org_action.arg().index() != act.arg().index()
                conflict_type = 3
                conflict_args = (s, org_action.ref_state().state(), org_action.arg(), act.arg())
                raise_error = False
                self._resolve_rr_conflict += 1
                if act.arg().index() > org_action.arg().index():
                    return  # 不接受在后面的产生式

            # 针对Reduce/Shift的情况
            if (org_action.action() == ACTION_REDUCE and act.action() == ACTION_GOTO) or \
                    (org_action.action() == ACTION_GOTO and act.action() == ACTION_REDUCE):
                if org_action.action() == ACTION_REDUCE:
                    reduce_action = org_action
                    shift_action = act
                else:
                    reduce_action = act
                    shift_action = org_action
                reduce_production = reduce_action.arg()  # type: Production
                shift_state = shift_action.arg()  # type: ExtendProductionSet
                assert isinstance(reduce_production, Production)
                assert isinstance(shift_state, ExtendProductionSet)
                assert shift_action.ref_symbol() == s
                assert s.type() != SYMBOL_NON_TERMINAL  # 非终结符不可能出现SR冲突
                conflict_type = 2
                conflict_args = (s, org_action.ref_state().state(), reduce_production)

                accept_reduce = None
                raise_error = False

                # 首先尝试算符优先
                # 语法规则保证定义了结合性时必然定义了算符优先级，对于没有定义算符优先级的表达式/符号不会通过算符优先方式解决
                if s.type() == SYMBOL_TERMINAL and s.precedence() > 0 and reduce_production.precedence() > 0:
                    # 如果优先级一致，则考虑结合性
                    if s.precedence() == reduce_production.precedence():
                        # 找到Reduce产生式的符号获取结合性
                        reduce_symbol = None
                        for i in range(len(reduce_production) - 1, -1, -1):
                            if reduce_production[i].type() == SYMBOL_TERMINAL:
                                reduce_symbol = reduce_production[i]
                                break
                        assert reduce_symbol is not None

                        if reduce_symbol.associativity() == ASSOC_NONE or s.associativity() == ASSOC_NONE:
                            # 没有结合性，报错
                            raise_error = True
                        elif reduce_symbol.associativity() == ASSOC_UNDEF or s.associativity() == ASSOC_UNDEF:
                            # 未定义结合性，回退到Shift优先规则
                            pass
                        elif reduce_symbol.associativity() != s.associativity():
                            # 结合性不一致，报错
                            raise_error = True
                        else:
                            # 结合性一致，按照结合性解决SR冲突
                            assert reduce_symbol.associativity() == s.associativity()

                            # 如果为左结合，则采取Reduce操作，否则采取Shift操作
                            if s.associativity() == ASSOC_LEFT:
                                accept_reduce = True
                            else:
                                assert s.associativity() == ASSOC_RIGHT
                                accept_reduce = False
                            self._resolve_sr_conflict_by_prec += 1
                    else:  # 优先级不一致，选择优先级高的进行reduce/shift
                        if reduce_production.precedence() > s.precedence():
                            accept_reduce = True
                        else:
                            accept_reduce = False
                        self._resolve_sr_conflict_by_prec += 1

                # 在算符优先也没有解决的情况下，优先使用Shift规则
                if (accept_reduce is None) and (not raise_error):
                    accept_reduce = False
                    self._resolve_sr_conflict_by_shift += 1

                # 最终决定是否接受覆盖
                if accept_reduce is not None:
                    assert not raise_error
                    if accept_reduce and reduce_action == org_action:
                        return
                    elif not accept_reduce and reduce_action == act:
                        return

            assert conflict_type != 0
            if raise_error:  # 未能解决冲突
                if conflict_type == 1:
                    raise GrammarError(f"Shift/shift conflict detected, symbol {repr(conflict_args[0])}, state: "
                                       f"{repr(conflict_args[1])}, shift state 1: {repr(conflict_args[2])}, "
                                       f"shift state 2: {repr(conflict_args[3])}")
                elif conflict_type == 2:
                    raise GrammarError(f"Shift/reduce conflict detected, state: {repr(conflict_args[1])}, "
                                       f"shift symbol: {repr(conflict_args[0])}, reduce production: "
                                       f"{repr(conflict_args[2])}")
                elif conflict_type == 3:
                    assert False  # Reduce/reduce冲突总能被解决
        self._actions[s][state] = act  # 覆盖状态

    def _process_lr1(self):
        # 以首个规则作为入口
        entry_production = Production(kEntrySymbol, [self._doc.productions()[0].left(), kEofSymbol], {})
        entry_production_ex = ExtendProduction(entry_production, 0, set())
        entry_item_set = self._closure(ExtendProductionSet({entry_production_ex}, -1))
        entry_item_set.set_state(0)  # 首个状态

        # 初始化状态
        next_state = 1
        states = {entry_item_set: entry_item_set.state()}  # type: Dict[ExtendProductionSet, int]
        q = [entry_item_set]  # type: List[ExtendProductionSet]

        # 计算动作表
        while len(q) > 0:
            state = q.pop(0)
            assert states[state] == state.state()

            # 填写规约动作
            for p in state:
                if p.pos() >= len(p.production()):
                    for x in p.lookahead():
                        action = Action(ACTION_REDUCE, p.production(), state, x, p)
                        self._populate_action(x, state.state(), action)

            # 计算Shift/Goto/Accept
            for x in self._extend_symbols:
                goto = self._goto(state, x)
                if len(goto) == 0:
                    continue
                if x == kEofSymbol:
                    for p in goto:
                        if p.pos() >= len(p.production()):
                            action = Action(ACTION_ACCEPT, None, state, x, p)
                            self._populate_action(x, state.state(), action)
                        else:
                            assert False  # 经由Eof推导出的状态只能是Reduce，不可能出现其他情况
                else:
                    if goto in states:
                        goto.set_state(states[goto])
                    else:
                        goto.set_state(next_state)
                        next_state += 1
                        states[goto] = goto.state()
                        q.append(goto)
                    assert goto.state() != -1
                    action = Action(ACTION_GOTO, goto, state, x, None)
                    self._populate_action(x, state.state(), action)
        self._max_state = next_state - 1

    def document(self):
        """
        获取原始语法文件
        :return: 文档对象
        """
        return self._doc

    def actions(self):
        """
        获取计算后的动作表
        :return: 动作转换表
        """
        return self._actions

    def max_state(self):
        """
        获取最大的状态ID
        """
        return self._max_state

    def printable_actions(self) -> str:
        """
        获取可打印动作表
        :return: 字符串结果
        """
        ret = []
        header = [None]  # 表头
        for s in self._doc.terminals():
            header.append(s)
        header.append(kEofSymbol)
        for s in self._doc.non_terminals():
            header.append(s)
        min_width = len(str(self._max_state)) + 1
        header_width = [max(min_width, len(s.id()) if s is not None else 0) for s in header]

        # 打印表头
        ret.append(" | ".join([header[i].id().rjust(header_width[i]) if header[i] is not None else
                               "".rjust(header_width[i]) for i in range(0, len(header))]))

        # 打印所有行
        for s in range(0, self._max_state + 1):
            empty = True
            data = []
            for i in range(0, len(header)):
                if i == 0:
                    data.append(str(s).rjust(header_width[i]))
                else:
                    if s in self._actions[header[i]]:
                        data.append(repr(self._actions[header[i]][s]).rjust(header_width[i]))
                        empty = False
                    else:
                        data.append("".rjust(header_width[i]))
            if not empty:
                ret.append(" | ".join(data))
        return "\n".join(ret)

    def process(self, mode):
        """
        处理语法
        :param mode: 语法模式
        """
        self._reset_actions()
        if mode == GRAMMAR_MODE_LR1:
            self._process_lr1()
        else:
            assert mode == GRAMMAR_MODE_LALR
            # TODO
            raise NotImplementedError()

    def resolve_stat(self) -> Tuple[int, int, int]:
        return self._resolve_rr_conflict, self._resolve_sr_conflict_by_prec, self._resolve_sr_conflict_by_shift

# ---------------------------------------- 模板渲染器 ----------------------------------------
# 见 https://github.com/9chu/et-py


class TemplateNode:
    def __init__(self, parent):
        self.parent = parent
        self.nodes = []

    def render(self, context):
        pass


class TemplateForNode(TemplateNode):
    def __init__(self, parent, identifier, expression):
        TemplateNode.__init__(self, parent)
        self.identifier = identifier
        self.expression = expression

    def render(self, context):
        result = eval(self.expression, None, context)
        origin = context[self.identifier] if self.identifier in context else None
        for i in result:
            context[self.identifier] = i
            yield iter(self.nodes)
        if origin:
            context[self.identifier] = origin


class TemplateIfNode(TemplateNode):
    def __init__(self, parent, expression):
        TemplateNode.__init__(self, parent)
        self.expression = expression
        self.true_branch = self.nodes

    def render(self, context):
        test = eval(self.expression, None, context)
        if test:
            yield iter(self.true_branch)


class TemplateIfElseNode(TemplateNode):
    def __init__(self, parent, if_node):  # extent from IfNode
        TemplateNode.__init__(self, parent)
        self.expression = if_node.expression
        self.true_branch = if_node.true_branch
        self.false_branch = self.nodes

    def render(self, context):
        test = eval(self.expression, None, context)
        if test:
            yield iter(self.true_branch)
        else:
            yield iter(self.false_branch)


class TemplateExpressionNode(TemplateNode):
    def __init__(self, parent, expression):
        TemplateNode.__init__(self, parent)
        self.expression = expression

    def render(self, context):
        return eval(self.expression, None, context)


class TextConsumer:
    def __init__(self, text):
        self._text = text
        self._len = len(text)
        self._pos = 0
        self._line = 1
        self._row = 0

    def get_pos(self):
        return self._pos

    def get_line(self):
        return self._line

    def get_row(self):
        return self._row

    def read(self):
        if self._pos >= self._len:
            return '\0'
        ch = self._text[self._pos]
        self._pos += 1
        self._row += 1
        if ch == '\n':
            self._line += 1
            self._row = 0
        return ch

    def peek(self, advance=0):
        if self._pos + advance >= self._len:
            return '\0'
        return self._text[self._pos + advance]

    def substr(self, begin, end):
        return self._text[begin:end]


class TemplateParser:
    OUTER_TOKEN_LITERAL = 1
    OUTER_TOKEN_EXPRESS = 2

    RESERVED = ["and", "as", "assert", "break", "class", "continue", "def", "del", "elif", "else", "except", "exec",
                "finally", "for", "from", "global", "if", "import", "in", "is", "lambda", "not", "or", "pass", "print",
                "raise", "return", "try", "while", "with", "yield"]

    def __init__(self, text):
        self._text = text
        self._consumer = TextConsumer(text)

    @staticmethod
    def _is_starting_by_new_line(text):
        for i in range(0, len(text)):
            ch = text[i:i + 1]
            if ch == '\n':
                return True
            elif not ch.isspace():
                break
        return False

    @staticmethod
    def _is_ending_by_new_line(text):
        for i in range(len(text) - 1, -1, -1):
            ch = text[i:i + 1]
            if ch == '\n':
                return True
            elif not ch.isspace():
                break
        return False

    @staticmethod
    def _trim_left_until_new_line(text):
        for i in range(0, len(text)):
            ch = text[i:i+1]
            if ch == '\n':
                return text[i+1:]
            elif not ch.isspace():
                break
        return text

    @staticmethod
    def _trim_right_until_new_line(text):
        for i in range(len(text) - 1, -1, -1):
            ch = text[i:i+1]
            if ch == '\n':
                return text[0:i+1]  # save right \n
            elif not ch.isspace():
                break
        return text

    @staticmethod
    def _parse_blank(consumer):
        while consumer.peek().isspace():  # 跳过所有空白
            consumer.read()

    @staticmethod
    def _parse_identifier(consumer):
        ch = consumer.peek()
        if not (ch.isalpha() or ch == '_'):
            return ""
        chars = [consumer.read()]  # ch
        ch = consumer.peek()
        while ch.isalnum() or ch == '_':
            chars.append(consumer.read())  # ch
            ch = consumer.peek()
        return "".join(chars)

    @staticmethod
    def _parse_inner(content, line, row):
        """内层解析函数
        考虑到表达式解析非常费力不讨好，这里采用偷懒方式进行。
        表达式全部交由python自行解决，匹配仅匹配开头，此外不处理注释（意味着不能在表达式中包含注释内容）。
        当满足 for <identifier> in <...> 时产生 for节点
        当满足 if <...> 时产生 if节点
        当满足 elif <...> 时产生 elif节点
        当满足 else 时产生 else节点
        当满足 end 时产生 end节点
        :param content: 内层内容
        :param line: 起始行
        :param row: 起始列
        :return: 节点名称, 表达式部分, 可选的Identifier
        """
        consumer = TextConsumer(content)
        TemplateParser._parse_blank(consumer)
        operator = TemplateParser._parse_identifier(consumer)
        identifier = None
        if operator == "for":
            TemplateParser._parse_blank(consumer)
            identifier = TemplateParser._parse_identifier(consumer)
            if identifier == "" or (identifier in TemplateParser.RESERVED):
                raise ParseError("Identifier expected", consumer.get_line() + line - 1,
                                 consumer.get_row() + row if consumer.get_line() == 1 else consumer.get_row())
            TemplateParser._parse_blank(consumer)
            if TemplateParser._parse_identifier(consumer) != "in":
                raise ParseError("Keyword 'in' expected", consumer.get_line() + line - 1,
                                 consumer.get_row() + row if consumer.get_line() == 1 else consumer.get_row())
            TemplateParser._parse_blank(consumer)
            expression = content[consumer.get_pos():]
            if expression == "":
                raise ParseError("Expression expected", consumer.get_line() + line - 1,
                                 consumer.get_row() + row if consumer.get_line() == 1 else consumer.get_row())
        elif operator == "if" or operator == "elif":
            TemplateParser._parse_blank(consumer)
            expression = content[consumer.get_pos():]
            if expression == "":
                raise ParseError("Expression expected", consumer.get_line() + line - 1,
                                 consumer.get_row() + row if consumer.get_line() == 1 else consumer.get_row())
        elif operator == "end" or operator == "else":
            TemplateParser._parse_blank(consumer)
            expression = content[consumer.get_pos():]
            if expression != '':
                raise ParseError("Unexpected content", consumer.get_line() + line - 1,
                                 consumer.get_row() + row if consumer.get_line() == 1 else consumer.get_row())
        else:
            operator = ""
            expression = content
        return operator, expression.strip(), identifier

    def _parse_outer(self):
        """外层解析函数
        将输入拆分成字符串(Literal)和表达式(Expression)两个组成。
        遇到'{%'开始解析Expression，在解析Expression时允许使用'%%'转义，即'%%'->'%'，这使得'%%>'->'%>'而不会结束表达式。
        :return: 类型, 内容, 起始行, 起始列
        """
        begin = self._consumer.get_pos()
        end = begin  # [begin, end)
        begin_line = self._consumer.get_line()
        begin_row = self._consumer.get_row()
        ch = self._consumer.peek()
        while ch != '\0':
            if ch == '{':
                ahead = self._consumer.peek(1)
                if ahead == '%':
                    if begin != end:
                        return TemplateParser.OUTER_TOKEN_LITERAL, self._consumer.substr(begin, end), begin_line, \
                               begin_row
                    self._consumer.read()  # {
                    self._consumer.read()  # %
                    begin_line = self._consumer.get_line()
                    begin_row = self._consumer.get_row()
                    chars = []
                    while True:
                        ch = self._consumer.read()
                        if ch == '\0':
                            raise ParseError("Unexpected eof", self._consumer.get_line(), self._consumer.get_row())
                        elif ch == '%':
                            if self._consumer.peek() == '}':  # '%}'
                                self._consumer.read()
                                return TemplateParser.OUTER_TOKEN_EXPRESS, "".join(chars), begin_line, begin_row
                            elif self._consumer.peek() == '%':  # '%%' -> '%'
                                self._consumer.read()
                        chars.append(ch)
            self._consumer.read()
            ch = self._consumer.peek()
            end = self._consumer.get_pos()
        return TemplateParser.OUTER_TOKEN_LITERAL, self._consumer.substr(begin, end), begin_line, begin_row

    @staticmethod
    def _trim_empty_line(result):
        state = 0
        left = None  # 需要剔除右边的元素
        for i in range(0, len(result)):
            cur = result[i]
            p = result[i - 1] if i != 0 else None
            n = result[i + 1] if i != len(result) - 1 else None
            if state == 0:
                # 当前是表达式，且上一个是文本
                if cur[0] == TemplateParser.OUTER_TOKEN_EXPRESS:
                    if p is None or (p[0] == TemplateParser.OUTER_TOKEN_LITERAL and
                                     TemplateParser._is_ending_by_new_line(p[1])):
                        left = i - 1 if p else None
                        state = 1
            if state == 1:
                if n is None or (n[0] == TemplateParser.OUTER_TOKEN_LITERAL and
                                 TemplateParser._is_starting_by_new_line(n[1])):
                    right = i + 1 if n else None
                    if left is not None:
                        result[left] = (result[left][0],
                                        TemplateParser._trim_right_until_new_line(result[left][1]),
                                        result[left][2],
                                        result[left][3])
                    if right is not None:
                        result[right] = (result[right][0],
                                         TemplateParser._trim_left_until_new_line(result[right][1]),
                                         result[right][2],
                                         result[right][3])
                    state = 0
                elif cur[0] != TemplateParser.OUTER_TOKEN_EXPRESS:  # 行中有其他文本，不进行剔除
                    state = 0

    def process(self):
        root = []  # 根
        nodes = []  # 未闭合节点队列
        outer_results = []
        while True:  # 为了剔除空行，需要先解析完所有的根元素做预处理
            ret = self._parse_outer()
            if ret[0] == TemplateParser.OUTER_TOKEN_LITERAL and ret[1] == "":  # EOF
                break
            outer_results.append(ret)
        TemplateParser._trim_empty_line(outer_results)
        for i in outer_results:
            (t, content, line, row) = i
            back = None if len(nodes) == 0 else nodes[len(nodes) - 1]
            if t == TemplateParser.OUTER_TOKEN_LITERAL:
                root.append(content) if back is None else back.nodes.append(content)
            else:
                assert t == TemplateParser.OUTER_TOKEN_EXPRESS
                (operator, expression, identifier) = self._parse_inner(content, line, row)
                if operator == "for":
                    node = TemplateForNode(back, identifier, expression)
                    root.append(node) if back is None else back.nodes.append(node)
                    nodes.append(node)
                elif operator == "if":
                    node = TemplateIfNode(back, expression)
                    root.append(node) if back is None else back.nodes.append(node)
                    nodes.append(node)
                elif operator == "else":
                    if not isinstance(back, TemplateIfNode):
                        raise ParseError("Unexpected else branch", line, row)
                    node = TemplateIfElseNode(back.parent, back)
                    # 从root或者父节点中删除back
                    if back.parent is None:
                        assert root[len(root) - 1] == back
                        root.pop()
                        root.append(node)
                    else:
                        parent_nodes = back.parent.nodes
                        assert parent_nodes[len(parent_nodes) - 1] == back
                        parent_nodes.pop()
                        parent_nodes.append(node)
                    # 升级并取代
                    nodes.pop()
                    nodes.append(node)
                elif operator == "elif":
                    if not isinstance(back, TemplateIfNode):
                        raise ParseError("Unexpected elif branch", line, row)
                    closed_else = TemplateIfElseNode(back.parent, back)
                    # 从root或者父节点中删除back
                    if back.parent is None:
                        assert root[len(root) - 1] == back
                        root.pop()
                        root.append(closed_else)
                    else:
                        parent_nodes = back.parent.nodes
                        assert parent_nodes[len(parent_nodes) - 1] == back
                        parent_nodes.pop()
                        parent_nodes.append(closed_else)
                    node = TemplateIfNode(closed_else, expression)
                    closed_else.nodes.append(node)
                    # 取代
                    nodes.pop()
                    nodes.append(node)
                elif operator == "end":
                    if back is None:
                        raise ParseError("Unexpected block end", line, row)
                    nodes.pop()  # 完成一个节点
                else:
                    assert operator == ""
                    node = TemplateExpressionNode(back, expression)
                    root.append(node) if back is None else back.nodes.append(node)
        if len(nodes) != 0:
            raise ParseError("Unclosed block", self._consumer.get_line(), self._consumer.get_row())
        return root


def render_template(template, **context):
    p = TemplateParser(template)
    root = p.process()
    output = []
    stack = [iter(root)]
    while stack:
        node = stack.pop()
        if isinstance(node, str):
            output.append(node)
        elif isinstance(node, TemplateExpressionNode):
            output.append(str(node.render(context)))
        elif isinstance(node, TemplateNode):
            stack.append(node.render(context))
        else:
            new_node = next(node, None)
            if new_node is not None:
                stack.append(node)
                stack.append(new_node)
    return "".join(output)


# ---------------------------------------- 代码生成 ----------------------------------------


def generate_code(header_template: str, source_template: str, analyzer: GrammarAnalyzer, header_filename: str):
    # 对所有符号进行整理，下标即最终的符号ID
    symbols = [kEofSymbol]
    tmp = list(analyzer.document().terminals())
    tmp.sort(key=lambda s: s.id())
    symbols.extend(tmp)
    token_cnt = len(symbols)
    tmp = list(analyzer.document().non_terminals())
    tmp.sort(key=lambda s: s.id())
    symbols.extend(tmp)

    # 生成token信息
    token_info = []
    for i in range(0, token_cnt):
        assert symbols[i].type() == SYMBOL_TERMINAL or symbols[i].type() == SYMBOL_EOF
        token_info.append({
            "id": i,
            "c_name": "_" if symbols[i] == kEofSymbol else symbols[i].id(),
            "raw": symbols[i]
        })

    # 生成映射表
    symbol_mapping = {}
    for i in range(0, len(symbols)):
        s = symbols[i]
        symbol_mapping[s] = i

    # 生成产生式信息
    production_info = []
    for i in range(0, len(analyzer.document().productions())):
        p = analyzer.document().productions()[i]
        assert i == p.index()
        production_info.append({
            "symbol": symbol_mapping[p.left()],
            "count": len(p),
            "raw": p
        })

    # 生成动作表
    actions = []
    state_remap_id_to_state_id = {}
    state_id_to_state_remap_id = {}
    offset = 0
    state_cnt = 0
    for i in range(0, analyzer.max_state() + 1):
        empty_state = True
        if i in analyzer.actions()[kEofSymbol]:
            empty_state = False
        else:
            for s in analyzer.document().symbols():
                if i in analyzer.actions()[s]:
                    empty_state = False
                    break
        if empty_state:
            offset += 1
        else:
            assert i not in state_id_to_state_remap_id
            assert (i - offset) not in state_remap_id_to_state_id
            state_id_to_state_remap_id[i] = i - offset
            state_remap_id_to_state_id[i - offset] = i
            state_cnt += 1
    for i in range(0, state_cnt):
        action = []
        for j in range(0, len(symbols)):
            s = symbols[j]
            one_action = [0, 0]
            state = state_remap_id_to_state_id[i]
            if state in analyzer.actions()[s]:
                act = analyzer.actions()[s][state]
                one_action[0] = act.action()
                if one_action == ACTION_GOTO:
                    one_action[1] = state_id_to_state_remap_id[act.arg().state()]
                elif one_action == ACTION_REDUCE:
                    assert analyzer.document().productions()[act.arg().index()] == act.arg()
                    one_action[1] = act.arg().index()
            action.append(one_action)
        actions.append(action)

    # 生成C++类型
    token_types = []
    need_monostate = False
    for s in analyzer.document().terminals():
        if s.replace() is None:
            need_monostate = True
        else:
            assert s.replace().strip() == s.replace()
            assert s.replace() != "std::monostate"
            if s.replace() not in token_types:
                token_types.append(s.replace())
    token_types.sort()
    if need_monostate or len(token_types) == 0:
        token_types.insert(0, "std::monostate")
    production_types = []
    need_monostate = False
    for s in analyzer.document().non_terminals():
        if s.replace() is None:
            need_monostate = True
        else:
            assert s.replace().strip() == s.replace()
            assert s.replace() != "std::monostate"
            if s.replace() not in production_types:
                production_types.append(s.replace())
    production_types.sort()
    if need_monostate or len(production_types) == 0:
        production_types.insert(0, "std::monostate")

    # generate the context
    args = analyzer.document().generator_args() or {}
    context = {
        "namespace": args.get("namespace", None),
        "class_name": args.get("class_name", "Parser"),
        "includes": args.get("includes", []),
        "symbols": symbols,
        "token_info": token_info,
        "token_types": token_types,
        "production_info": production_info,
        "production_types": production_types,
        "actions": actions,
        "header_filename": header_filename,
    }

    header_src = render_template(header_template, **context)
    source_src = render_template(source_template, **context)
    return header_src, source_src

# ---------------------------------------- Main ----------------------------------------


CPP_HEADER_TPL = """/**
 * @file
 * @date {% datetime.date.today() %}
 *
 * Auto generated code by 9chu/parser_gen.
 */
#pragma once
#include <cstdint>
#include <vector>
#include <variant>

{% for f in includes %}
#include "{% f %}"
{% end %}

{% if namespace is None %}
// namespace {
{% else %}
namespace {% namespace %}
{
{% end %}
    class {% class_name %}
    {
    public:
        enum class ParseResult
        {
            NotKnown = 0,
            Accepted = 1,
            Rejected = 2,
        };
        
        enum class Tokens
        {
            {% for t in token_info %}
            {% t["c_name"] %} = {% t["id"] %},
            {% end %}
        };
        
        using TokenValues = std::variant<
            {% for i in range(0, len(token_types)) %}
            {% token_types[i] %}{% if i != len(token_types) - 1 %},{% end %}
            {% end %}
            >;
            
        using ProductionValues = std::variant<
            {% for i in range(0, len(production_types)) %}
            {% production_types[i] %}{% if i != len(production_types) - 1 %},{% end %}
            {% end %}
            >;
        
        using UnionValues = std::variant<TokenValues, ProductionValues>;
        
    public:
        {% class_name %}();
        
    public:
        ParseResult operator()(Tokens token, const TokenValues& value);
        void Reset()noexcept;
        
    private:
        std::vector<uint32_t> m_stStack;
        std::vector<UnionValues> m_stValueStack;
    };
{% if namespace is None %}
// }
{% else %}
}
{% end %}
"""

CPP_SOURCE_TPL = """/**
 * @file
 * @date {% datetime.date.today() %}
 *
 * Auto generated code by 9chu/parser_gen.
 */
#include "{% header_filename %}"

#include <cassert>

{% if namespace is not None %}
using namespace {% namespace %};
{% end %}

#define ACTION_ERROR 0
#define ACTION_ACCEPT 1
#define ACTION_GOTO 2
#define ACTION_REDUCE 3

namespace {
    {% for idx in range(0, len(production_info)) %}
    {% class_name %}::ProductionValues Reduce{% idx %}(const std::vector<{% class_name %}::UnionValues>& stack_)
    {
        // binding values
        assert(stack_.size() >= {% len(production_info[idx]["raw"]) %});
        {% for pos in production_info[idx]["raw"].binding() %}
        const auto& {% production_info[idx]["raw"].binding()[pos] %} =
        {% if production_info[idx]["raw"][pos].type() == 2 %}
            std::get<{% production_info[idx]["raw"][pos].replace() %}>(
                std::get<{% class_name %}::ProductionValues>(stack_[stack_.size() - {% len(production_info[idx]["raw"]) + pos %}]));
        {% else %}
            std::get<{% production_info[idx]["raw"][pos].replace() %}>(
                std::get<{% class_name %}::TokenValues>(stack_[stack_.size() - {% len(production_info[idx]["raw"]) + pos %}]));{% end %}
        {% end %}
        
        // user code
        {% if production_info[idx]["raw"].left().replace() is not None %}
        auto ret = [&]() {
            {% production_info[idx]["raw"].replace().strip() %}
        }();
        return {% class_name %}::ProductionValues { std::move(ret) };
        {% else %}
        {% production_info[idx]["raw"].replace() %}
        return {% class_name %}::ProductionValues {};
        {% end %}
    }
    
    {% end %}
}

using ReduceFunction = {% class_name %}::ProductionValues(*)(const std::vector<{% class_name %}::UnionValues>&);

struct ProductionInfo
{
    uint32_t NonTerminal;
    uint32_t SymbolCount;
    ReduceFunction Callback;
};

struct ActionInfo
{
    uint8_t Action;
    uint32_t Arg;
};

static const ProductionInfo kProductions[{% len(production_info) %}] = {
    {% for i in range(0, len(production_info)) %}
    { {% production_info[i]["symbol"] %}, {% production_info[i]["count"] %}, ::Reduce{% i %} },
    {% end %}
};

static const ActionInfo kActions[{% len(actions) %}][{% len(symbols) %}] = {
    {% for action in actions %}
    { {% for act in action %}{ {% act[0] %}, {% act[1] %} },{% end %} },
    {% end %}
};

{% class_name %}::{% class_name %}()
{
    Reset();
}

{% class_name %}::ParseResult {% class_name %}::operator()(Tokens token, const TokenValues& value)
{
    assert(!m_stStack.empty());
    assert(static_cast<uint32_t>(token) < {% len(token_info) %});
    
    const ActionInfo& act = kActions[m_stStack.back()][static_cast<uint32_t>(token)];
    if (act.Action == ACTION_ACCEPT)
    {
        Reset();
        return ParseResult::Accepted;
    }
    else if (act.Action == ACTION_ERROR)
    {
        Reset();
        return ParseResult::Rejected;
    }
    else if (act.Action == ACTION_GOTO)
    {
        m_stStack.push_back(static_cast<uint32_t>(token));
        m_stStack.push_back(act.Arg);
        assert(m_stStack.back() < {% len(actions) %});
        
        m_stValueStack.push_back(value);
    }
    else
    {
        assert(act.Action == ACTION_REDUCE);
        assert(act.Arg < {% len(production_info) %});
        
        const ProductionInfo& info = kProductions[act.Arg];
        auto val = info.Callback(m_stValueStack);
        
        assert(m_stStack.size() >= info.SymbolCount * 2);
        m_stStack.resize(m_stStack.size() - info.SymbolCount * 2);
        
        assert(m_stValueStack.size() >= info.SymbolCount);
        m_stValueStack.resize(m_stValueStack.size() - info.SymbolCount);
        
        m_stValueStack.emplace_back(std::move(val));
        assert(!m_stStack.empty());
        
        const ActionInfo& act2 = kActions[m_stStack.back()][info.NonTerminal];
        if (act2.Action == ACTION_GOTO)
        {
            m_stStack.push_back(info.NonTerminal);
            m_stStack.push_back(act2.Arg);
        }
        else
        {
            assert(false);
            Reset();
            return ParseResult::Rejected;
        }
    }
    
    return ParseResult::NotKnown;
}

void {% class_name %}::Reset()noexcept
{
    m_stStack.clear();
    m_stValueStack.clear();
    
    // initial state
    m_stStack.push_back(0);
}
"""


def main():
    parser = argparse.ArgumentParser(description="A LR(1)/LALR(1) parser generator for C++17.")
    parser.add_argument("--header-file", type=str, help="Output header filename", default="Parser.hpp")
    parser.add_argument("--source-file", type=str, help="Output source filename", default="Parser.cpp")
    parser.add_argument("-o", "--output-dir", type=str, help="Output directory", default="./")
    parser.add_argument("--header-template", type=str, help="User defined header template", default="")
    parser.add_argument("--source-template", type=str, help="User defined source template", default="")
    parser.add_argument("--lalr", type=bool, help="Set to LALR(1) mode", default=False)
    parser.add_argument("--print-actions", type=bool, help="Print action table", default=False)
    parser.add_argument("grammar-filename", help="Grammar filename")
    args = parser.parse_args()

    doc = GrammarDocument()
    doc.parse(args.grammar_filename)

    analyzer = GrammarAnalyzer(doc)
    analyzer.process(GRAMMAR_MODE_LALR if args.lalr else GRAMMAR_MODE_LR1)

    if args.print_actions:
        print(analyzer.printable_actions())

    resolve_rr_cnt, resolve_sr_by_prec_cnt, resolve_sr_by_shift_cnt = analyzer.resolve_stat()
    sys.stderr.write(f"Reduce/Reduce conflict resolved count: {resolve_rr_cnt}\n")
    sys.stderr.write(f"Shift/Reduce conflict resolved count (by Operator Precedence): {resolve_sr_by_prec_cnt}\n")
    sys.stderr.write(f"Shift/Reduce conflict resolved count (by Shift Priority): {resolve_sr_by_shift_cnt}\n")

    header_tpl_content = CPP_HEADER_TPL
    source_tpl_content = CPP_SOURCE_TPL
    if args.header_template != "":
        with open(args.header_template, "r", encoding="utf-8") as f:
            header_tpl_content = f.read()
    if args.source_template != "":
        with open(args.source_template, "r", encoding="utf-8") as f:
            source_tpl_content = f.read()
    header_output, source_output = generate_code(header_tpl_content, source_tpl_content, analyzer, args.header_file)
    with open(os.path.join(args.output_dir, args.header_file), "w", encoding="utf-8") as f:
        f.write(header_output)
    with open(os.path.join(args.output_dir, args.source_file), "w", encoding="utf-8") as f:
        f.write(source_output)


if __name__ == "__main__":
    main()
