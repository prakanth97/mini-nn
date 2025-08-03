#include <cassert>
#include <cctype>
#include <cstddef>
#include <cstdio>
#include <cstdlib>
#include <string>

#include "lexer.h"

FILE *pFile; // Global file pointer for the lexer

//===----------------------------------------------------------------------===//
// Lexer
//===----------------------------------------------------------------------===//

static std::string globalLexeme; // Filled in if IDENT
static int lineNo, columnNo;

const std::string TOKEN::getIdentifierStr() const {
  if (type != IDENT) {
    fprintf(stderr, "%d:%d Error: %s\n", lineNo, columnNo,
            "getIdentifierStr called on non-IDENT token");
    exit(2);
  }
  return lexeme;
}

int TOKEN::getIntVal() const {
  if (type != INT_LIT) {
    fprintf(stderr, "%d:%d Error: %s\n", lineNo, columnNo,
            "getIntVal called on non-INT_LIT token");
    exit(2);
  }
  return strtod(lexeme.c_str(), nullptr);
}

float TOKEN::getFloatVal() const {
  if (type != FLOAT_LIT) {
    fprintf(stderr, "%d:%d Error: %s\n", lineNo, columnNo,
            "getFloatVal called on non-FLOAT_LIT token");
    exit(2);
  }
  return strtof(lexeme.c_str(), nullptr);
}

bool TOKEN::getBoolVal() const {
  if (type != BOOL_LIT) {
    fprintf(stderr, "%d:%d Error: %s\n", lineNo, columnNo,
            "getBoolVal called on non-BOOL_LIT token");
    exit(2);
  }
  return (lexeme == "true");
}

static TOKEN returnTok(std::string lexVal, int tok_type) {
  TOKEN return_tok;
  return_tok.lexeme = lexVal;
  return_tok.type = tok_type;
  return_tok.lineNo = lineNo;
  return_tok.columnNo = columnNo - lexVal.length() - 1;
  return return_tok;
}

// Read file line by line -- or look for \n and if found add 1 to line number
// and reset column number to 0
/// gettok - Return the next token from standard input.
TOKEN getTok() {

  static int LastChar = ' ';
  static int NextChar = ' ';

  // Skip any whitespace.
  while (isspace(LastChar)) {
    if (LastChar == '\n' || LastChar == '\r') {
      lineNo++;
      columnNo = 1;
    }
    LastChar = getc(pFile);
    columnNo++;
  }

  if (isalpha(LastChar) ||
      (LastChar == '_')) { // identifier: [a-zA-Z_][a-zA-Z_0-9]*
    globalLexeme = LastChar;
    columnNo++;

    while (isalnum((LastChar = getc(pFile))) || (LastChar == '_')) {
      globalLexeme += LastChar;
      columnNo++;
    }

		return getKeywordOrIdentToken(); // Check if it's a keyword
  }

  if (LastChar == '=') {
    NextChar = getc(pFile);
    if (NextChar == '=') { // EQ: ==
      LastChar = getc(pFile);
      columnNo += 2;
      return returnTok("==", EQ);
    } else {
      LastChar = NextChar;
      columnNo++;
      return returnTok("=", ASSIGN);
    }
  }

  if (LastChar == '{') {
    LastChar = getc(pFile);
    columnNo++;
    return returnTok("{", LBRA);
  }
  if (LastChar == '}') {
    LastChar = getc(pFile);
    columnNo++;
    return returnTok("}", RBRA);
  }
  if (LastChar == '(') {
    LastChar = getc(pFile);
    columnNo++;
    return returnTok("(", LPAR);
  }
  if (LastChar == ')') {
    LastChar = getc(pFile);
    columnNo++;
    return returnTok(")", RPAR);
  }
  if (LastChar == ';') {
    LastChar = getc(pFile);
    columnNo++;
    return returnTok(";", SC);
  }
  if (LastChar == ',') {
    LastChar = getc(pFile);
    columnNo++;
    return returnTok(",", COMMA);
  }
	if (LastChar == '[') {
		LastChar = getc(pFile);
		columnNo++;
		return returnTok("[", LBOX);
	}
	if (LastChar == ']') {
		LastChar = getc(pFile);
		columnNo++;
		return returnTok("]", RBOX);
	}

  if (isdigit(LastChar) || LastChar == '.') { // Number: [0-9]+.
    std::string NumStr;

    if (LastChar == '.') { // Floatingpoint Number: .[0-9]+
      do {
        NumStr += LastChar;
        LastChar = getc(pFile);
        columnNo++;
      } while (isdigit(LastChar));

      //   FloatVal = strtof(NumStr.c_str(), nullptr);
      return returnTok(NumStr, FLOAT_LIT);
    } else {
      do { // Start of Number: [0-9]+
        NumStr += LastChar;
        LastChar = getc(pFile);
        columnNo++;
      } while (isdigit(LastChar));

      if (LastChar == '.') { // Floatingpoint Number: [0-9]+.[0-9]+)
        do {
          NumStr += LastChar;
          LastChar = getc(pFile);
          columnNo++;
        } while (isdigit(LastChar));

        // FloatVal = strtof(NumStr.c_str(), nullptr);
        return returnTok(NumStr, FLOAT_LIT);
      } else { // Integer : [0-9]+
        // IntVal = strtod(NumStr.c_str(), nullptr);
        return returnTok(NumStr, INT_LIT);
      }
    }
  }

	if (LastChar == '#') {
		// Comment: #.*\n
		do {
			LastChar = getc(pFile);
			columnNo++;
		} while (LastChar != EOF && LastChar != '\n' && LastChar != '\r');

		return getTok(); // Recurse to get the next token after the comment
	}

  // Check for end of file.  Don't eat the EOF.
  if (LastChar == EOF) {
    columnNo++;
    return returnTok("0", EOF_TOK);
  }

  // Otherwise, just return the character as its ascii value.
  int ThisChar = LastChar;
  std::string s(1, ThisChar);
  LastChar = getc(pFile);
  columnNo++;
  return returnTok(s, int(ThisChar));
}

static bool isBuiltinFunc(const std::string &lexeme) {
	return (lexeme == "bool" || lexeme == "int" || lexeme == "float" ||
					lexeme == "void" || lexeme == "transpose" || lexeme == "matmul" ||
					lexeme == "add" || lexeme == "relu" || lexeme == "sigmoid" ||
					lexeme == "tanh" || lexeme == "softmax");
}

static bool isLayerType(const std::string &lexeme) {
	return (lexeme == "dense" || lexeme == "conv1d" || lexeme == "conv2d");
}

TOKEN getKeywordOrIdentToken() {	
  if (globalLexeme == "func")
    return returnTok("func", FUNC_TOK);
	if (globalLexeme == "int")
		return returnTok("int", INT_TOK);
	if (globalLexeme == "bool")
		return returnTok("bool", BOOL_TOK);
	if (globalLexeme == "float")
		return returnTok("float", FLOAT_TOK);
	if (globalLexeme == "void")
		return returnTok("void", VOID_TOK);
	if (globalLexeme == "bool")
		return returnTok("bool", BOOL_TOK);
	if (globalLexeme == "return")
		return returnTok("return", RETURN);
	if (globalLexeme == "true")
		return returnTok("true", BOOL_LIT);
	if (globalLexeme == "false") 	
		return returnTok("false", BOOL_LIT);
	if (globalLexeme == "layer")
		return returnTok("layer", LAYER_TYPE_TOK);
	if (isLayerType(globalLexeme))
		return returnTok(globalLexeme.c_str(), LAYER_TOK);
	if (isBuiltinFunc(globalLexeme))
		return returnTok(globalLexeme.c_str(), BUILTIN_FUNC_TOK);

	return returnTok(globalLexeme.c_str(), IDENT); // If not a keyword, return as IDENT
}
