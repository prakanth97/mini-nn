#include "ast.h"
#include "lexer.h"
#include <deque>
#include <memory>
#include <stdio.h>
#include <vector>

class Parser {
public:
  TOKEN CurToken;
  std::deque<TOKEN> tok_buffer;

  TOKEN getNextToken() {
    if (tok_buffer.empty()) {
      tok_buffer.push_back(getTok());
    }

    TOKEN temp = tok_buffer.front();
    tok_buffer.pop_front();
    return CurToken = temp;
  }

  void pushBackToken(TOKEN tok) { tok_buffer.push_front(tok); }

  // Main parsing method
  std::unique_ptr<Program> parseProgram();

  // Grammar rule methods
  std::vector<std::unique_ptr<Statement>> parseStatementList();
  void
  parseStatementListPrime(std::vector<std::unique_ptr<Statement>> &statements);

  std::unique_ptr<Statement> parseStatement();
  std::unique_ptr<LayerDefinition> parseLayerDefinition();
  std::unique_ptr<FunctionDefinition> parseFunctionDefinition();

  std::vector<std::unique_ptr<Statement>> parseFunctionStatementList();
  void parseFunctionStatementListPrime(
      std::vector<std::unique_ptr<Statement>> &statements);

  std::unique_ptr<Statement> parseFunctionStatement();
  std::unique_ptr<Assignment> parseAssignment();
  std::unique_ptr<ReturnStatement> parseReturnStatement();

  std::unique_ptr<Expression> parseExpression();
  std::unique_ptr<Expression>
  parseExpressionSuffix(const std::string &identifier);

  std::vector<std::pair<std::unique_ptr<Type>, std::string>>
  parseParameterList();
  void parseParameterListPrime(
      std::vector<std::pair<std::unique_ptr<Type>, std::string>> &params);

  std::vector<std::unique_ptr<Expression>> parseArgumentList();
  void parseArgumentListPrime(std::vector<std::unique_ptr<Expression>> &args);

  std::unique_ptr<Type> parseType();
  void parseTypeSuffix(std::unique_ptr<Type> &type);

  std::unique_ptr<NumberLiteral> parseNumber();

  // Helper methods
  void expectToken(int expectedType, const std::string &errorMsg);
  void skipCommentText();

  // Error handling
  void error(const std::string &message);

private:
  void parseTopLevel(); // Keep for compatibility
};
