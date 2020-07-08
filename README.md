# parser_gen

一个小巧的`LR(1)/LALR(1)`解析器生成工具，适用于`C++17`或更高。

Working in progress.

## TODO

- [ ] 完成LALR支持

## 快速开始

Calculator.p:

```
term Plus assoc(left) prec(1);  // +
term Minus assoc(left) prec(1);  // -
term Multiply assoc(left) prec(2);  // *
term Division assoc(left) prec(2);  // /
term LeftParen;  // (
term RightParen;  // )
term LiteralNumber {% int %};

nonterm exp {% int %};

grammar {
    exp -> LiteralNumber(value) {% return value; %};
    exp -> LeftParen exp(exp) RightParen {% return exp; %};
    exp -> Minus exp(rhs) prec(10) {% return -rhs; %};
    exp -> exp(lhs) Plus exp(rhs) {% return lhs + rhs; %};
    exp -> exp(lhs) Minus exp(rhs) {% return lhs - rhs; %};
    exp -> exp(lhs) Multiply exp(rhs) {% return lhs * rhs; %};
    exp -> exp(lhs) Division exp(rhs) {% return lhs / rhs; %};
};

generator {%
    {
        "class_name": "CalculatorParser"
    }
%};
```

Main.cpp:

```c++
#include <tuple>
#include <variant>
#include <iostream>

#include "CalculatorParser.hpp"

class Tokenizer {
public:
    Tokenizer(const char* buffer)
        : m_pBuffer(buffer) {}
public:
    std::tuple<CalculatorParser::TokenTypes, CalculatorParser::TokenValues> Advance() {
        using TokenTypes = CalculatorParser::TokenTypes;
        using TokenValues = CalculatorParser::TokenValues;
        while (true) {
            if (*m_pBuffer == '\0')
                return { TokenTypes::_, TokenValues {} };

            char c;
            switch (c = *(m_pBuffer++)) {
                case '+': return { TokenTypes::Plus, TokenValues {} };
                case '-': return { TokenTypes::Minus, TokenValues {} };
                case '*': return { TokenTypes::Multiply, TokenValues {} };
                case '/': return { TokenTypes::Division, TokenValues {} };
                case '(': return { TokenTypes::LeftParen, TokenValues {} };
                case ')': return { TokenTypes::RightParen, TokenValues {} };
                case ' ':
                case '\t':
                case '\n':
                case '\r':
                    continue;
                default:
                    if (c >= '0' && c <= '9') {
                        int ret = (c - '0');
                        while (*m_pBuffer >= '0' && *m_pBuffer <= '9')
                            ret = ret * 10 + (*(m_pBuffer++) - '0');
                        return { TokenTypes::LiteralNumber, TokenValues { ret } };
                    }
                    else
                        throw std::runtime_error("Bad input");
            }
        }
    }
private:
    const char* m_pBuffer;
};

int main() {
    try {
        while (std::cin) {
            std::string input;
            std::getline(std::cin, input);

            Tokenizer tokenizer(input.c_str());
            CalculatorParser parser;
            while (true) {
                auto [t, v] = tokenizer.Advance();

                auto ret = parser(t, v);
                if (ret == CalculatorParser::ParseResult::Rejected)
                    throw std::runtime_error("Parse error");
                else if (ret == CalculatorParser::ParseResult::Accepted)
                    std::cout << parser.Result() << std::endl;

                if (t == CalculatorParser::TokenTypes::_)
                    break;
            }
        };
    }
    catch (const std::exception& ex) {
        std::cerr << ex.what() << std::endl;
        return 1;
    }
    return 0;
}
```

Build:

```bash
./parser_gen.py --header-file CalculatorParser.hpp --source-file CalculatorParser.cpp Calculator.p
g++ CalculatorParser.cpp Main.cpp -std=c++17 -o calculator
```

Run it:

```bash
./calculator
```

## 特性

- 生成可重入代码
- 不污染命名空间
- 用户驱动接口

## 语法规则文件

语法规则文件由四部分声明构成：
- 终结符
- 非终结符
- 规则
- 生成器参数

### 终结符

终结符使用下述方式声明：

```
  term 标识符 {% 替换 %} ;
```

其中，标识符用于指定终结符的名称，可以由非数字开头的若干数字、字母或者下划线构成（下同），需要注意的是单独的`_`会被识别为关键词。

替换部分应当填写一个C/C++类型，当语法制导翻译遇到一个标识符时可以给出对应的C/C++类型的值供用户代码使用。

若替换部分留空，则该标识符的值不可在翻译过程中被使用。

此外，为了支撑算符优先冲突解决规则，可以在标识符后面使用关键字`assoc`和`prec`来指定左结合或右结合以及对应的优先级，例如：

```
  term minus assoc(left) prec(1) {% Tokenizer::Token %};
```

其中`assoc`可以接`left`、`right`或者`none`，表明左结合、右结合或者无结合性。

其中`prec`用于指定算符优先级，算符优先级高的表达式会在`移进/规约`冲突中被优先选择。

在解决冲突时，如果发现算符无结合性则会产生错误，若解决冲突的任意一方不指定结合性或优先级，则会按照其他规约规则自动解决冲突。

此外，算符优先冲突解决规则仅适用于诸如：`Exp op Exp`的表达式，其中`op`是一个非终结符。

### 非终结符

非终结符使用下述方式声明：

```
  nonterm 标识符 {% 替换 %};
```

具体规则和终结符一致，但是不可以声明结合性或者优先级，其他内容不再赘述。

### 语法规则

声明完终结符和非终结符后可以声明语法规则，举例如下：

```
  grammar {
    Exp -> Exp(lhs) plus Exp(rhs) {% return Ast::BinExp(lhs, rhs, Ast::BinOp::Plus); %};
    Exp -> Exp(lhs) minus Exp(rhs) {% return Ast::BinExp(lhs, rhs, Ast::BinOp::Minus); %};
  }
```

语法规则定义在`grammar`块中，一个产生式具备下述形式：

```
  非终结符 -> 符号1 ( 标识符1 ) 符号2 ( 标识符2 ) ... {% 替换 %} ;
```

其中，非终结符指示从哪个终结符推导而来，整个产生式在规约后将会具备该终结符对应的类型。

`符号1..n`指示产生式的构成，每个符号可以接一个标识符，将会在生成代码中使用符号对应的类型捕获值给解析器代码使用。

需要注意，首条规则被作为入口规则产生文法。此外如果产生式不规约任何符号，需要使用特殊的语法来声明：

```
  非终结符 -> _ {% 替换 %};
```

另外，为了支持单目运算符的特殊优先级，产生式本身可以指定一个独立的优先级，例如：

```
  grammar {
    UnaryExp -> minus Exp(rhs) prec(10) {% ... %};
  }
```

此时，`prec`必须在产生式末尾，当生成器在解决`BinExp`和`UnaryExp`的冲突时会优先匹配`UnaryExp`。

### 代码生成参数

在完成上述定义后，你可以使用 Json 来向代码生成器传递参数，这些参数会被用于在模板中替换对应的变量：

```
  generator {%
    {
      "namespace": "Test",
      "class_name": "MyParser",
      "includes": [
        "Ast.hpp"
      ]
    }
  %}
```

### 附录：关键词表
  
```
  _ term nonterm grammar generator assoc prec left right none
```

### 附录：规约/移进冲突解决规则

- 下述规则被依次用于解决规约/移进冲突：
  - 尝试使用算符优先和结合性规则进行解决；
  - 采取移进规则解决；
- 下述规则被依次用于解决规约/规约冲突：
  - 依照生成式的定义顺序解决，先定义的生成式会先被用于解决冲突；

## License

MIT License
