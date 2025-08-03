#include <string>
#include <vector>
#include <memory>

// Forward declarations
class Statement;
class Expression;
class Type;

// Base AST node
class ASTNode {
public:
    virtual ~ASTNode() = default;
    virtual std::string toString(int indent = 0) const = 0;
    
protected:
    std::string getIndent(int level) const {
        return std::string(level * 2, ' ');
    }
};

// Program node
class Program : public ASTNode {
public:
    std::vector<std::unique_ptr<Statement>> statements;
    Program(std::vector<std::unique_ptr<Statement>> stmts) 
        : statements(std::move(stmts)) {}
    std::string toString(int indent = 0) const override;
};

// Base Statement class
class Statement : public ASTNode {
public:
    virtual ~Statement() = default;
    virtual std::string toString(int indent = 0) const override = 0;
};

// Layer definition
class LayerDefinition : public Statement {
public:
    std::string name;
    std::string layerType;
    std::vector<std::unique_ptr<Expression>> parameters;
    bool hasLayerKeyword;
    
    LayerDefinition(const std::string& n, const std::string& type, 
                   std::vector<std::unique_ptr<Expression>> params, bool hasKeyword)
        : name(n), layerType(type), parameters(std::move(params)), hasLayerKeyword(hasKeyword) {}

    std::string toString(int indent = 0) const override;
};

// Function definition
class FunctionDefinition : public Statement {
public:
    std::string name;
    std::vector<std::pair<std::unique_ptr<Type>, std::string>> parameters;
    std::unique_ptr<Type> returnType;
    std::vector<std::unique_ptr<Statement>> body;
    
    FunctionDefinition(const std::string& n, 
                      std::vector<std::pair<std::unique_ptr<Type>, std::string>> params,
                      std::unique_ptr<Type> retType,
                      std::vector<std::unique_ptr<Statement>> bodyStmts)
        : name(n), parameters(std::move(params)), returnType(std::move(retType)), 
          body(std::move(bodyStmts)) {}
    
    std::string toString(int indent = 0) const override;
};

// Assignment statement
class Assignment : public Statement {
public:
    std::string variable;
    std::unique_ptr<Expression> value;
    
    Assignment(const std::string& var, std::unique_ptr<Expression> val)
        : variable(var), value(std::move(val)) {}
    
    std::string toString(int indent = 0) const override;
};

// Return statement
class ReturnStatement : public Statement {
public:
    std::unique_ptr<Expression> value;
    
    ReturnStatement(std::unique_ptr<Expression> val) : value(std::move(val)) {}

    std::string toString(int indent = 0) const override; 
};

// Base Expression class
class Expression : public ASTNode {
public:
    virtual ~Expression() = default;
    virtual std::string toString(int indent = 0) const override = 0; 
};

// Identifier expression
class Identifier : public Expression {
public:
    std::string name;
    Identifier(const std::string& n) : name(n) {}
    
    std::string toString(int indent = 0) const override;
};

// Function call expression
class FunctionCall : public Expression {
public:
    std::string functionName;
    std::vector<std::unique_ptr<Expression>> arguments;
    bool isBuiltin;
    
    FunctionCall(const std::string& name, std::vector<std::unique_ptr<Expression>> args, bool builtin)
        : functionName(name), arguments(std::move(args)), isBuiltin(builtin) {}
    
    std::string toString(int indent = 0) const override;
};

// Number literal
class NumberLiteral : public Expression {
public:
    std::string value;
    bool isFloat;
    
    NumberLiteral(const std::string& val, bool isFloating) 
        : value(val), isFloat(isFloating) {}
    
    std::string toString(int indent = 0) const override;
};

// Type class
class Type : public ASTNode {
public:
    std::string primitiveType;
    std::vector<std::string> dimensions;
    
    Type(const std::string& primitive) : primitiveType(primitive) {}
    Type(const std::string& primitive, std::vector<std::string> dims) 
        : primitiveType(primitive), dimensions(std::move(dims)) {}
    
    std::string toString(int indent = 0) const override; 
};
