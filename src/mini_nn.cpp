#include <cstdio>
#include <iostream>

#include "parser/parser.h"

int main(int argc, char** argv) {
  if (argc == 2) {
    pFile = fopen(argv[1], "r");
    if (pFile == NULL)
      perror("Error opening file");
  } else {
    std::cout << "Usage: ./code InputFile\n";
    return 1;
  }

  Parser parser;
  auto program = parser.parseProgram();
  fclose(pFile);
  std::cout << program->toString() << std::endl;
  return 0;
}
