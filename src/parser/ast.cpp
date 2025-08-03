#include "ast.h"

std::string Program::toString(int indent) const {
    std::string result = getIndent(indent) + "Program {\n";
    for (const auto& stmt : statements) {
        result += stmt->toString(indent + 1) + "\n";
    }
    result += getIndent(indent) + "}";
    return result;
}

// LayerDefinition toString implementation
std::string LayerDefinition::toString(int indent) const {
    std::string result = getIndent(indent) + "LayerDefinition {\n";
    result += getIndent(indent + 1) + "hasKeyword: " + (hasLayerKeyword ? "true" : "false") + "\n";
    result += getIndent(indent + 1) + "name: \"" + name + "\"\n";
    result += getIndent(indent + 1) + "type: \"" + layerType + "\"\n";
    result += getIndent(indent + 1) + "parameters: [\n";
    for (const auto& param : parameters) {
        result += param->toString(indent + 2) + "\n";
    }
    result += getIndent(indent + 1) + "]\n";
    result += getIndent(indent) + "}";
    return result;
}

// FunctionDefinition toString implementation
std::string FunctionDefinition::toString(int indent) const {
    std::string result = getIndent(indent) + "FunctionDefinition {\n";
    result += getIndent(indent + 1) + "name: \"" + name + "\"\n";
    result += getIndent(indent + 1) + "parameters: [\n";
    for (const auto& param : parameters) {
        if (param.first) {
            result += getIndent(indent + 2) + "(" + param.first->toString(0) + ", \"" + param.second + "\")\n";
        } else {
            result += getIndent(indent + 2) + "(number, \"" + param.second + "\")\n";
        }
    }
    result += getIndent(indent + 1) + "]\n";
    result += getIndent(indent + 1) + "returnType: " + returnType->toString(0) + "\n";
    result += getIndent(indent + 1) + "body: [\n";
    for (const auto& stmt : body) {
        result += stmt->toString(indent + 2) + "\n";
    }
    result += getIndent(indent + 1) + "]\n";
    result += getIndent(indent) + "}";
    return result;
}

// Assignment toString implementation
std::string Assignment::toString(int indent) const {
    std::string result = getIndent(indent) + "Assignment {\n";
    result += getIndent(indent + 1) + "variable: \"" + variable + "\"\n";
    result += getIndent(indent + 1) + "value: " + value->toString(indent + 1) + "\n";
    result += getIndent(indent) + "}";
    return result;
}

// ReturnStatement toString implementation
std::string ReturnStatement::toString(int indent) const {
    std::string result = getIndent(indent) + "ReturnStatement {\n";
    result += getIndent(indent + 1) + "value: " + value->toString(indent + 1) + "\n";
    result += getIndent(indent) + "}";
    return result;
}

// Identifier toString implementation
std::string Identifier::toString(int indent) const {
    return getIndent(indent) + "Identifier: \"" + name + "\"";
}

// FunctionCall toString implementation
std::string FunctionCall::toString(int indent) const {
    std::string result = getIndent(indent) + "FunctionCall {\n";
    result += getIndent(indent + 1) + "name: \"" + functionName + "\"\n";
    result += getIndent(indent + 1) + "isBuiltin: " + (isBuiltin ? "true" : "false") + "\n";
    result += getIndent(indent + 1) + "arguments: [\n";
    for (const auto& arg : arguments) {
        result += arg->toString(indent + 2) + "\n";
    }
    result += getIndent(indent + 1) + "]\n";
    result += getIndent(indent) + "}";
    return result;
}

// NumberLiteral toString implementation
std::string NumberLiteral::toString(int indent) const {
    return getIndent(indent) + "NumberLiteral: " + value + " (" + (isFloat ? "float" : "int") + ")";
}

// Type toString implementation
std::string Type::toString(int indent) const {
    std::string result = getIndent(indent) + primitiveType;
    for (const auto& dim : dimensions) {
        result += "[" + dim + "]";
    }
    return result;
}
