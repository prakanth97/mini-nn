#include <memory>
#include <string>
#include <vector>

// Forward declarations
class Statement;
class Expression;
class Type;

// Enum for LLVM-style RTTI
enum ASTNodeKind {
  NK_Statement,
  NK_LayerDefinition,
  NK_FunctionDefinition,
  NK_Assignment,
  NK_ReturnStatement,
  NK_Expression,
  NK_Identifier,
  NK_FunctionCall,
  NK_NumberLiteral,
  NK_Type
};

// Base AST node
class ASTNode {
private:
  const ASTNodeKind kind;

public:
  ASTNodeKind getKind() const { return kind; }
  
  ASTNode(ASTNodeKind K) : kind(K) {}
  virtual ~ASTNode() = default;
  virtual std::string toString(int indent = 0) const = 0;

protected:
  std::string getIndent(int level) const { return std::string(level * 2, ' '); }
};

// Program node
class Program : public ASTNode {
public:
  std::vector<std::unique_ptr<Statement>> statements;
  Program(std::vector<std::unique_ptr<Statement>> stmts)
      : ASTNode(NK_Statement), statements(std::move(stmts)) {}
  std::string toString(int indent = 0) const override;
};

// Base Statement class
class Statement : public ASTNode {
public:
  Statement(ASTNodeKind K) : ASTNode(K) {}
  virtual ~Statement() = default;
  virtual std::string toString(int indent = 0) const override = 0;
  
  static bool classof(const ASTNode *N) {
    return N->getKind() >= NK_Statement && N->getKind() <= NK_ReturnStatement;
  }
};

// Layer definition
class LayerDefinition : public Statement {
public:
  std::string name;
  std::string layerType;
  std::vector<std::unique_ptr<Expression>> parameters;
  bool hasLayerKeyword;

  LayerDefinition(const std::string &n, const std::string &type,
                  std::vector<std::unique_ptr<Expression>> params,
                  bool hasKeyword)
      : Statement(NK_LayerDefinition), name(n), layerType(type), parameters(std::move(params)),
        hasLayerKeyword(hasKeyword) {}

  std::string toString(int indent = 0) const override;
  
  static bool classof(const ASTNode *N) {
    return N->getKind() == NK_LayerDefinition;
  }
};

// Function definition
class FunctionDefinition : public Statement {
public:
  std::string name;
  std::vector<std::pair<std::unique_ptr<Type>, std::string>> parameters;
  std::unique_ptr<Type> returnType;
  std::vector<std::unique_ptr<Statement>> body;

  FunctionDefinition(
      const std::string &n,
      std::vector<std::pair<std::unique_ptr<Type>, std::string>> params,
      std::unique_ptr<Type> retType,
      std::vector<std::unique_ptr<Statement>> bodyStmts)
      : Statement(NK_FunctionDefinition), name(n), parameters(std::move(params)), returnType(std::move(retType)),
        body(std::move(bodyStmts)) {}

  std::string toString(int indent = 0) const override;
  
  static bool classof(const ASTNode *N) {
    return N->getKind() == NK_FunctionDefinition;
  }
};

// Assignment statement
class Assignment : public Statement {
public:
  std::string variable;
  std::unique_ptr<Expression> value;

  Assignment(const std::string &var, std::unique_ptr<Expression> val)
      : Statement(NK_Assignment), variable(var), value(std::move(val)) {}

  std::string toString(int indent = 0) const override;
  
  static bool classof(const ASTNode *N) {
    return N->getKind() == NK_Assignment;
  }
};

// Return statement
class ReturnStatement : public Statement {
public:
  std::unique_ptr<Expression> value;

  ReturnStatement(std::unique_ptr<Expression> val) : Statement(NK_ReturnStatement), value(std::move(val)) {}

  std::string toString(int indent = 0) const override;
  
  static bool classof(const ASTNode *N) {
    return N->getKind() == NK_ReturnStatement;
  }
};

// Base Expression class
class Expression : public ASTNode {
public:
  Expression(ASTNodeKind K) : ASTNode(K) {}
  virtual ~Expression() = default;
  virtual std::string toString(int indent = 0) const override = 0;
  
  static bool classof(const ASTNode *N) {
    return N->getKind() >= NK_Expression && N->getKind() <= NK_NumberLiteral;
  }
};

// Identifier expression
class Identifier : public Expression {
public:
  std::string name;
  Identifier(const std::string &n) : Expression(NK_Identifier), name(n) {}

  std::string toString(int indent = 0) const override;
  
  static bool classof(const ASTNode *N) {
    return N->getKind() == NK_Identifier;
  }
};

// Function call expression
class FunctionCall : public Expression {
public:
  std::string functionName;
  std::vector<std::unique_ptr<Expression>> arguments;
  bool isBuiltin;

  FunctionCall(const std::string &name,
               std::vector<std::unique_ptr<Expression>> args, bool builtin)
      : Expression(NK_FunctionCall), functionName(name), arguments(std::move(args)), isBuiltin(builtin) {}

  std::string toString(int indent = 0) const override;
  
  static bool classof(const ASTNode *N) {
    return N->getKind() == NK_FunctionCall;
  }
};

// Number literal
class NumberLiteral : public Expression {
public:
  std::string value;
  bool isFloat;

  NumberLiteral(const std::string &val, bool isFloating)
      : Expression(NK_NumberLiteral), value(val), isFloat(isFloating) {}

  std::string toString(int indent = 0) const override;
  
  static bool classof(const ASTNode *N) {
    return N->getKind() == NK_NumberLiteral;
  }
};

// Type class
class Type : public ASTNode {
public:
  std::string primitiveType;
  std::vector<std::string> dimensions;

  Type(const std::string &primitive) : ASTNode(NK_Type), primitiveType(primitive) {}
  Type(const std::string &primitive, std::vector<std::string> dims)
      : ASTNode(NK_Type), primitiveType(primitive), dimensions(std::move(dims)) {}

  std::string toString(int indent = 0) const override;
  
  static bool classof(const ASTNode *N) {
    return N->getKind() == NK_Type;
  }
};
