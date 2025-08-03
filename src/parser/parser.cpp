#include "parser.h"
#include <iostream>
#include <stdexcept>

void Parser::parseTopLevel() {
  auto program = parseProgram();
}

std::unique_ptr<Program> Parser::parseProgram() {
  getNextToken(); // Initialize CurToken
  auto statements = parseStatementList();
  return std::make_unique<Program>(std::move(statements));
}

std::vector<std::unique_ptr<Statement>> Parser::parseStatementList() {
  std::vector<std::unique_ptr<Statement>> statements;

  auto stmt = parseStatement();
  if (stmt) {
    statements.push_back(std::move(stmt));
  }

  parseStatementListPrime(statements);
  return statements;
}

void Parser::parseStatementListPrime(
    std::vector<std::unique_ptr<Statement>> &statements) {
  // Check if we can parse another statement
  if (CurToken.type == '#' || CurToken.type == LAYER_TYPE_TOK ||
      CurToken.type == IDENT || CurToken.type == FUNC_TOK) {

    auto stmt = parseStatement();
    if (stmt) {
      statements.push_back(std::move(stmt));
    }
    parseStatementListPrime(statements);
  }
  // else epsilon (do nothing)
}

std::unique_ptr<Statement> Parser::parseStatement() {
  if (CurToken.type == '#') {
    return nullptr;
  } else if (CurToken.type == LAYER_TYPE_TOK || CurToken.type == IDENT) {
    return parseLayerDefinition();
  } else if (CurToken.type == FUNC_TOK) {
    return parseFunctionDefinition();
  } else if (CurToken.type == EOF_TOK) {
    return nullptr;
  } else {
    error("Expected comment, layer definition, or function definition");
    return nullptr;
  }
}

std::unique_ptr<LayerDefinition> Parser::parseLayerDefinition() {
  bool hasLayerKeyword = false;
  std::string name;

  if (CurToken.type == LAYER_TYPE_TOK) {
    hasLayerKeyword = true;
    getNextToken(); // consume "layer"

    if (CurToken.type != IDENT) {
      error("Expected identifier after 'layer'");
      return nullptr;
    }
    name = CurToken.getIdentifierStr();
    getNextToken();
  } else if (CurToken.type == IDENT) {
    name = CurToken.getIdentifierStr();
    getNextToken();
  } else {
    error("Expected 'layer' keyword or identifier");
    return nullptr;
  }

  expectToken('=', "Expected '=' in layer definition");

  if (CurToken.type != LAYER_TOK) {
    error("Expected layer type (dense, conv1d, or conv2d)");
    return nullptr;
  }

  std::string layerType = CurToken.lexeme;
  getNextToken();

  expectToken('(', "Expected '(' after layer type");

  auto parameters =
      parseArgumentList(); // Reuse argument list parsing for parameters

  expectToken(')', "Expected ')' after parameter list");
  expectToken(';', "Expected ';' after layer definition");

  return std::make_unique<LayerDefinition>(
      name, layerType, std::move(parameters), hasLayerKeyword);
}

std::unique_ptr<FunctionDefinition> Parser::parseFunctionDefinition() {
  expectToken(FUNC_TOK, "Expected 'func' keyword");

  if (CurToken.type != IDENT) {
    error("Expected function name");
    return nullptr;
  }

  std::string funcName = CurToken.getIdentifierStr();
  getNextToken();

  expectToken('(', "Expected '(' after function name");

  auto parameters = parseParameterList();

  expectToken(')', "Expected ')' after parameter list");
  expectToken(RETURN, "Expected 'return' keyword");

  auto returnType = parseType();

  expectToken('{', "Expected '{' to start function body");

  auto body = parseFunctionStatementList();

  expectToken('}', "Expected '}' to end function body");

  return std::make_unique<FunctionDefinition>(
      funcName, std::move(parameters), std::move(returnType), std::move(body));
}

std::vector<std::unique_ptr<Statement>> Parser::parseFunctionStatementList() {
  std::vector<std::unique_ptr<Statement>> statements;

  auto stmt = parseFunctionStatement();
  if (stmt) {
    statements.push_back(std::move(stmt));
  }

  parseFunctionStatementListPrime(statements);
  return statements;
}

void Parser::parseFunctionStatementListPrime(
    std::vector<std::unique_ptr<Statement>> &statements) {
  // Check if we can parse another function statement
  if (CurToken.type == IDENT || CurToken.type == RETURN) {
    auto stmt = parseFunctionStatement();
    if (stmt) {
      statements.push_back(std::move(stmt));
    }
    parseFunctionStatementListPrime(statements);
  }
  // else epsilon (do nothing)
}

std::unique_ptr<Statement> Parser::parseFunctionStatement() {
  if (CurToken.type == IDENT) {
    return parseAssignment();
  } else if (CurToken.type == RETURN) {
    return parseReturnStatement();
  } else {
    error("Expected assignment or return statement");
    return nullptr;
  }
}

std::unique_ptr<Assignment> Parser::parseAssignment() {
  if (CurToken.type != IDENT) {
    error("Expected identifier in assignment");
    return nullptr;
  }

  std::string varName = CurToken.getIdentifierStr();
  getNextToken();

  expectToken('=', "Expected '=' in assignment");

  auto expr = parseExpression();

  if (CurToken.type == ';') {
    getNextToken();
  }

  return std::make_unique<Assignment>(varName, std::move(expr));
}

std::unique_ptr<ReturnStatement> Parser::parseReturnStatement() {
  expectToken(RETURN, "Expected 'return' keyword");

  auto expr = parseExpression();

  expectToken(';', "Expected ';' after return statement");

  return std::make_unique<ReturnStatement>(std::move(expr));
}

std::unique_ptr<Expression> Parser::parseExpression() {
	if (CurToken.type == INT_LIT) {
		auto number = parseNumber();
		return std::make_unique<NumberLiteral>(number->value, false);
	} else if (CurToken.type == FLOAT_LIT) {
		auto number = parseNumber();
		return std::make_unique<NumberLiteral>(number->value, true);
	} else if (CurToken.type == IDENT) {
    std::string identifier = CurToken.getIdentifierStr();
    getNextToken();
    return parseExpressionSuffix(identifier);
  } else if (CurToken.type == BUILTIN_FUNC_TOK) {
    std::string funcName = CurToken.lexeme;
    getNextToken();

    expectToken('(', "Expected '(' after builtin function");

    auto args = parseArgumentList();

    expectToken(')', "Expected ')' after argument list");

    return std::make_unique<FunctionCall>(funcName, std::move(args), true);
  } else {
    error("Expected identifier or builtin function in expression");
    return nullptr;
  }
}

std::unique_ptr<Expression> Parser::parseExpressionSuffix(const std::string &identifier) {
  if (CurToken.type == '(') {
    getNextToken(); // consume '('

    auto args = parseArgumentList();

    expectToken(')', "Expected ')' after argument list");

    return std::make_unique<FunctionCall>(identifier, std::move(args), false);
  } else {
    // epsilon - just return the identifier
    return std::make_unique<Identifier>(identifier);
  }
}

std::vector<std::pair<std::unique_ptr<Type>, std::string>> Parser::parseParameterList() {
  std::vector<std::pair<std::unique_ptr<Type>, std::string>> parameters;

  if (CurToken.type == INT_TOK || CurToken.type == FLOAT_TOK ||
      CurToken.type == BOOL_TOK || CurToken.type == INT_LIT ||
      CurToken.type == FLOAT_LIT) {

    if (CurToken.type == INT_LIT || CurToken.type == FLOAT_LIT) {
      // Parameter is just a number
      auto number = parseNumber();
      parameters.push_back(std::make_pair(nullptr, number->value));
    } else {
      // Parameter is type + identifier
      auto type = parseType();

      if (CurToken.type != IDENT) {
        error("Expected identifier after type in parameter");
        return parameters;
      }

      std::string paramName = CurToken.getIdentifierStr();
      getNextToken();

      parameters.push_back(std::make_pair(std::move(type), paramName));
    }

    parseParameterListPrime(parameters);
  }
  // else epsilon (empty parameter list)

  return parameters;
}

void Parser::parseParameterListPrime(
    std::vector<std::pair<std::unique_ptr<Type>, std::string>> &params) {
  if (CurToken.type == ',') {
    getNextToken(); // consume ','

    if (CurToken.type == INT_LIT || CurToken.type == FLOAT_LIT) {
      auto number = parseNumber();
      params.push_back(std::make_pair(nullptr, number->value));
    } else {
      auto type = parseType();

      if (CurToken.type != IDENT) {
        error("Expected identifier after type in parameter");
        return;
      }

      std::string paramName = CurToken.getIdentifierStr();
      getNextToken();

      params.push_back(std::make_pair(std::move(type), paramName));
    }

    parseParameterListPrime(params);
  }
  // else epsilon (do nothing)
}

std::vector<std::unique_ptr<Expression>> Parser::parseArgumentList() {
  std::vector<std::unique_ptr<Expression>> arguments;

	// TODO: This is wrong, arguments can int or identifier
  if (CurToken.type == INT_LIT || CurToken.type == FLOAT_LIT 
		|| CurToken.type == IDENT || CurToken.type == BUILTIN_FUNC_TOK) {
    auto expr = parseExpression();
    if (expr) {
      arguments.push_back(std::move(expr));
    }

    parseArgumentListPrime(arguments);
  }
  // else epsilon (empty argument list)

  return arguments;
}

void Parser::parseArgumentListPrime(
    std::vector<std::unique_ptr<Expression>> &args) {
  if (CurToken.type == ',') {
    getNextToken(); // consume ','

    auto expr = parseExpression();
    if (expr) {
      args.push_back(std::move(expr));
    }

    parseArgumentListPrime(args);
  }
  // else epsilon (do nothing)
}

std::unique_ptr<Type> Parser::parseType() {
  if (CurToken.type != INT_TOK && CurToken.type != FLOAT_TOK &&
      CurToken.type != BOOL_TOK) {
    error("Expected primitive type (int, float, or bool)");
    return nullptr;
  }

  std::string primitiveType = CurToken.lexeme;
  getNextToken();

  auto type = std::make_unique<Type>(primitiveType);
  parseTypeSuffix(type);

  return type;
}

void Parser::parseTypeSuffix(std::unique_ptr<Type> &type) {
  if (CurToken.type == '[') {
    getNextToken(); // consume '['

    auto number = parseNumber();
    type->dimensions.push_back(number->value);

    expectToken(']', "Expected ']' after array dimension");

    parseTypeSuffix(type); // Recursive call for multi-dimensional arrays
  }
  // else epsilon (do nothing)
}

std::unique_ptr<NumberLiteral> Parser::parseNumber() {
  if (CurToken.type == INT_LIT) {
    std::string value = CurToken.lexeme;
    getNextToken();
    return std::make_unique<NumberLiteral>(value, false);
  } else if (CurToken.type == FLOAT_LIT) {
    std::string value = CurToken.lexeme;
    getNextToken();
    return std::make_unique<NumberLiteral>(value, true);
  } else {
    error("Expected number literal");
    return nullptr;
  }
}

void Parser::expectToken(int expectedType, const std::string &errorMsg) {
  if (CurToken.type != expectedType) {
    error(errorMsg + ". Got: " + std::to_string(CurToken.type));
    return;
  }
  getNextToken();
}

void Parser::skipCommentText() {
  while (CurToken.type != '\n' && CurToken.type != EOF_TOK) {
    getNextToken();
  }
}

void Parser::error(const std::string &message) {
  std::cerr << "Parser Error at line " << CurToken.lineNo << ", column "
            << CurToken.columnNo << ": " << message << std::endl;
  throw std::runtime_error("Parser error: " + message);
}
