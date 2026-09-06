#include "clang/AST/ASTContext.h"
#include "clang/AST/DeclCXX.h"
#include "clang/AST/Expr.h"
#include "clang/AST/ParentMapContext.h"
#include "clang/AST/RecursiveASTVisitor.h"
#include "clang/AST/Stmt.h"
#include "clang/AST/StmtCXX.h"
#include "clang/ASTMatchers/ASTMatchFinder.h"
#include "clang/Basic/SourceManager.h"
#include "clang/Lex/Lexer.h"
#include "clang/Tooling/CommonOptionsParser.h"
#include "clang/Tooling/Tooling.h"
#include "llvm/ADT/SmallString.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/Support/CommandLine.h"
#include "llvm/Support/FileSystem.h"
#include "llvm/Support/JSON.h"
#include "llvm/Support/Path.h"
#include "llvm/Support/raw_ostream.h"

#include <algorithm>
#include <cstdint>
#include <fstream>
#include <iterator>
#include <map>
#include <optional>
#include <string>
#include <tuple>
#include <utility>
#include <vector>

namespace {

using clang::ASTContext;
using clang::BinaryOperator;
using clang::CallExpr;
using clang::CompoundStmt;
using clang::CXXConstructorDecl;
using clang::DeclRefExpr;
using clang::Expr;
using clang::ForStmt;
using clang::FunctionDecl;
using clang::IfStmt;
using clang::RecursiveASTVisitor;
using clang::SourceLocation;
using clang::SourceManager;
using clang::Stmt;
using clang::StringLiteral;
using clang::VarDecl;
using clang::ast_matchers::cxxConstructorDecl;
using clang::ast_matchers::isDefinition;
using clang::ast_matchers::isExpansionInMainFile;
using clang::ast_matchers::MatchFinder;
using clang::ast_matchers::unless;

llvm::cl::OptionCategory RewriterCategory("skippy-stage-rewriter options");

llvm::cl::opt<std::string>
    SourceRoot("source-root",
               llvm::cl::desc("Root of the prepared llama.cpp source tree"),
               llvm::cl::value_desc("path"), llvm::cl::Required,
               llvm::cl::cat(RewriterCategory));

llvm::cl::opt<std::string> ReportPath(
    "report",
    llvm::cl::desc("Write the deterministic JSON report to this path"),
    llvm::cl::value_desc("path"), llvm::cl::Required,
    llvm::cl::cat(RewriterCategory));

llvm::cl::opt<std::string> LlamaCommit(
    "llama-commit",
    llvm::cl::desc("Exact llama.cpp commit represented by the source tree"),
    llvm::cl::value_desc("sha"), llvm::cl::Required,
    llvm::cl::cat(RewriterCategory));

llvm::cl::opt<bool>
    Apply("apply", llvm::cl::desc("Apply proven edits to the source tree"),
          llvm::cl::init(false), llvm::cl::cat(RewriterCategory));

struct Edit {
  std::string kind;
  std::string file;
  uint64_t offset = 0;
  uint64_t length = 0;
  std::string text;
};

struct Proof {
  std::string loop_var;
  std::string loop_start;
  std::string loop_end;
  std::string activation_in;
  std::string activation_out;
  bool embedding_owner = false;
  bool output_owner = false;
  std::vector<std::string> terminal_predicates;
  std::vector<std::string> nonlocal_exits;
  std::string execution_scope = "partitioned_decoder";
  std::vector<std::string> scope_evidence;
};

struct BuilderReport {
  std::string file;
  std::string constructor;
  unsigned line = 0;
  std::string verdict;
  std::string unsupported_reason;
  Proof proof;
  std::vector<Edit> edits;
};

std::string sourceText(clang::SourceRange range, const SourceManager &sm,
                       const clang::LangOptions &lang) {
  if (range.isInvalid() || range.getBegin().isMacroID() ||
      range.getEnd().isMacroID()) {
    return {};
  }
  return clang::Lexer::getSourceText(
             clang::CharSourceRange::getTokenRange(range), sm, lang)
      .str();
}

std::optional<std::string> referencedName(const Expr *expr) {
  if (expr == nullptr) {
    return std::nullopt;
  }
  expr = expr->IgnoreParenImpCasts();
  if (const auto *ref = llvm::dyn_cast<DeclRefExpr>(expr)) {
    return ref->getDecl()->getNameAsString();
  }
  if (const auto *member = llvm::dyn_cast<clang::MemberExpr>(expr)) {
    return member->getMemberDecl()->getNameAsString();
  }
  return std::nullopt;
}

const FunctionDecl *directCallee(const CallExpr *call) {
  return call == nullptr ? nullptr : call->getDirectCallee();
}

std::optional<std::pair<uint64_t, uint64_t>>
tokenRange(clang::SourceRange range, const SourceManager &sm,
           const clang::LangOptions &lang);

std::vector<const BinaryOperator *> assignmentsTo(const Stmt *root,
                                                   llvm::StringRef variable);

struct HyperconnectionPrelude {
  const VarDecl *carried_decl = nullptr;
  const BinaryOperator *repeat_assignment = nullptr;
  const Stmt *repeat_statement = nullptr;
  std::string width;
  std::string multiplicity;
  std::string tokens;
};

bool containsName(const Stmt *statement, llvm::StringRef target) {
  class Visitor final : public RecursiveASTVisitor<Visitor> {
  public:
    explicit Visitor(llvm::StringRef target) : target_(target) {}

    bool VisitDeclRefExpr(DeclRefExpr *ref) {
      found_ |= ref->getDecl()->getNameAsString() == target_;
      return !found_;
    }

    bool VisitMemberExpr(clang::MemberExpr *member) {
      found_ |= member->getMemberDecl()->getNameAsString() == target_;
      return !found_;
    }

    bool found() const { return found_; }

  private:
    llvm::StringRef target_;
    bool found_ = false;
  } visitor(target);
  visitor.TraverseStmt(const_cast<Stmt *>(statement));
  return visitor.found();
}

bool containsLayerBound(const Expr *expression) {
  class Visitor final : public RecursiveASTVisitor<Visitor> {
  public:
    bool VisitDeclRefExpr(DeclRefExpr *ref) {
      found_ |= llvm::StringRef(ref->getDecl()->getNameAsString())
                    .contains("n_layer");
      return !found_;
    }

    bool VisitMemberExpr(clang::MemberExpr *member) {
      found_ |= llvm::StringRef(member->getMemberDecl()->getNameAsString())
                    .contains("n_layer");
      return !found_;
    }

    bool found() const { return found_; }

  private:
    bool found_ = false;
  } visitor;
  visitor.TraverseStmt(const_cast<Expr *>(expression));
  return visitor.found();
}

std::string stableLoopEnd(const Expr *expression, const SourceManager &sm,
                          const clang::LangOptions &lang) {
  const Expr *normalized = expression->IgnoreParenImpCasts();
  if (const auto *reference = llvm::dyn_cast<DeclRefExpr>(normalized)) {
    if (const auto *variable = llvm::dyn_cast<VarDecl>(reference->getDecl())) {
      if (variable->getType().isConstQualified() && variable->hasInit()) {
        const std::string initializer =
            sourceText(variable->getInit()->getSourceRange(), sm, lang);
        if (!initializer.empty()) {
          return initializer;
        }
      }
    }
  }
  return sourceText(expression->getSourceRange(), sm, lang);
}

const Stmt *directChildContaining(const CompoundStmt *body, const Stmt *needle,
                                  const SourceManager &sm,
                                  const clang::LangOptions &lang) {
  const auto needle_range = tokenRange(needle->getSourceRange(), sm, lang);
  if (!needle_range) {
    return nullptr;
  }
  for (const Stmt *statement : body->body()) {
    const auto statement_range =
        tokenRange(statement->getSourceRange(), sm, lang);
    if (!statement_range) {
      continue;
    }
    if (statement_range->first <= needle_range->first &&
        statement_range->first + statement_range->second >=
            needle_range->first + needle_range->second) {
      return statement;
    }
  }
  return nullptr;
}

std::optional<std::string> assignedName(const CallExpr *call,
                                        ASTContext &context) {
  clang::DynTypedNode current = clang::DynTypedNode::create(*call);
  for (unsigned depth = 0; depth < 24; ++depth) {
    const auto parents = context.getParents(current);
    if (parents.size() != 1) {
      return std::nullopt;
    }
    const auto &parent = parents[0];
    if (const auto *binary = parent.get<BinaryOperator>()) {
      if (binary->isAssignmentOp()) {
        return referencedName(binary->getLHS());
      }
    }
    if (const auto *variable = parent.get<VarDecl>()) {
      return variable->getNameAsString();
    }
    if (parent.get<CompoundStmt>() != nullptr) {
      return std::nullopt;
    }
    current = parent;
  }
  return std::nullopt;
}

std::optional<uint64_t> fileOffset(SourceLocation location,
                                   const SourceManager &sm) {
  location = sm.getSpellingLoc(location);
  if (!location.isValid() || location.isMacroID() ||
      !sm.isWrittenInMainFile(location)) {
    return std::nullopt;
  }
  return sm.getFileOffset(location);
}

std::optional<std::pair<uint64_t, uint64_t>>
tokenRange(clang::SourceRange range, const SourceManager &sm,
           const clang::LangOptions &lang) {
  const auto begin = fileOffset(range.getBegin(), sm);
  const SourceLocation end_location = clang::Lexer::getLocForEndOfToken(
      sm.getSpellingLoc(range.getEnd()), 0, sm, lang);
  const auto end = fileOffset(end_location, sm);
  if (!begin || !end || *end < *begin) {
    return std::nullopt;
  }
  return std::pair<uint64_t, uint64_t>{*begin, *end - *begin};
}

std::string indentationAt(SourceLocation location, const SourceManager &sm) {
  location = sm.getSpellingLoc(location);
  if (!location.isValid() || !sm.isWrittenInMainFile(location)) {
    return {};
  }
  const char *data = sm.getCharacterData(location);
  const unsigned column = sm.getSpellingColumnNumber(location);
  std::string indent;
  for (unsigned i = 1; i < column; ++i) {
    const char ch = data[-static_cast<ptrdiff_t>(column - i)];
    if (ch != ' ' && ch != '\t') {
      return {};
    }
    indent.push_back(ch);
  }
  return indent;
}

class ExitVisitor final : public RecursiveASTVisitor<ExitVisitor> {
public:
  ExitVisitor(ASTContext &context, const ForStmt *target)
      : context_(context), target_(target) {}

  bool TraverseLambdaExpr(clang::LambdaExpr *) { return true; }

  bool VisitReturnStmt(clang::ReturnStmt *) {
    exits.emplace_back("return");
    return true;
  }

  bool VisitGotoStmt(clang::GotoStmt *) {
    exits.emplace_back("goto");
    return true;
  }

  bool VisitCXXThrowExpr(clang::CXXThrowExpr *) {
    exits.emplace_back("throw");
    return true;
  }

  bool VisitBreakStmt(clang::BreakStmt *statement) {
    if (targetsLayerLoop(statement, true)) {
      exits.emplace_back("break");
    }
    return true;
  }

  bool VisitContinueStmt(clang::ContinueStmt *statement) {
    if (targetsLayerLoop(statement, false)) {
      continues.push_back(statement);
    }
    return true;
  }

  std::vector<std::string> exits;
  std::vector<const clang::ContinueStmt *> continues;

private:
  bool targetsLayerLoop(const Stmt *statement, bool breakable) const {
    clang::DynTypedNode current = clang::DynTypedNode::create(*statement);
    for (unsigned depth = 0; depth < 48; ++depth) {
      const auto parents = context_.getParents(current);
      if (parents.size() != 1) {
        return false;
      }
      const auto &parent = parents[0];
      if (const auto *loop = parent.get<ForStmt>()) {
        return loop == target_;
      }
      if (parent.get<clang::WhileStmt>() != nullptr ||
          parent.get<clang::DoStmt>() != nullptr ||
          (breakable && parent.get<clang::SwitchStmt>() != nullptr)) {
        return false;
      }
      current = parent;
    }
    return false;
  }

  ASTContext &context_;
  const ForStmt *target_;
};

int layerLoopScore(const ForStmt *loop, llvm::StringRef activation,
                   const SourceManager &sm,
                   const clang::LangOptions &lang) {
  const auto *body = llvm::dyn_cast<CompoundStmt>(loop->getBody());
  const auto *condition = llvm::dyn_cast_or_null<BinaryOperator>(loop->getCond());
  if (body == nullptr || condition == nullptr) {
    return -1;
  }

  class Visitor final : public RecursiveASTVisitor<Visitor> {
  public:
    bool TraverseLambdaExpr(clang::LambdaExpr *) { return true; }

    bool VisitCallExpr(CallExpr *call) {
      const auto *callee = directCallee(call);
      if (callee == nullptr) {
        return true;
      }
      const std::string name = callee->getNameAsString();
      has_cvec |= name == "build_cvec";
      if (name == "cb" && call->getNumArgs() >= 2) {
        const auto *label = llvm::dyn_cast<StringLiteral>(
            call->getArg(1)->IgnoreParenImpCasts());
        if (label != nullptr) {
          const llvm::StringRef value = label->getString();
          has_layer_output |= value == "l_out" || value == "l_last";
        }
      }
      return true;
    }

    bool has_cvec = false;
    bool has_layer_output = false;
  } visitor;
  visitor.TraverseStmt(const_cast<CompoundStmt *>(body));

  int score = 0;
  score += visitor.has_cvec ? 16 : 0;
  score += visitor.has_layer_output ? 16 : 0;
  score += containsName(body, "layers") ? 4 : 0;
  score += assignmentsTo(body, activation).empty() ? 0 : 8;
  score += sourceText(condition->getRHS()->getSourceRange(), sm, lang) ==
                   "il_end"
               ? 2
               : 0;
  return score;
}

std::optional<std::string>
layerCarriedName(const CompoundStmt *body, llvm::StringRef embedding_activation,
                 ASTContext &context, const SourceManager &sm,
                 const clang::LangOptions &lang) {
  if (!assignmentsTo(body, embedding_activation).empty()) {
    return embedding_activation.str();
  }

  struct Candidate {
    uint64_t offset;
    std::string name;
  };
  class Visitor final : public RecursiveASTVisitor<Visitor> {
  public:
    Visitor(ASTContext &context, const SourceManager &sm,
            const clang::LangOptions &lang, std::vector<Candidate> &candidates)
        : context_(context), sm_(sm), lang_(lang), candidates_(candidates) {}

    bool TraverseLambdaExpr(clang::LambdaExpr *) { return true; }

    bool VisitCallExpr(CallExpr *call) {
      const auto *callee = directCallee(call);
      if (callee == nullptr) {
        return true;
      }
      const std::string callee_name = callee->getNameAsString();
      std::optional<std::string> name;
      if (callee_name == "build_cvec") {
        name = assignedName(call, context_);
        if (!name && call->getNumArgs() > 0) {
          name = referencedName(call->getArg(0));
        }
      } else if (callee_name == "cb" && call->getNumArgs() >= 2) {
        const auto *label = llvm::dyn_cast<StringLiteral>(
            call->getArg(1)->IgnoreParenImpCasts());
        if (label != nullptr &&
            (label->getString() == "l_out" ||
             label->getString() == "l_last")) {
          name = referencedName(call->getArg(0));
        }
      }
      const auto offset = fileOffset(call->getBeginLoc(), sm_);
      if (name && offset) {
        candidates_.push_back(Candidate{*offset, *name});
      }
      return true;
    }

  private:
    ASTContext &context_;
    const SourceManager &sm_;
    const clang::LangOptions &lang_;
    std::vector<Candidate> &candidates_;
  };

  std::vector<Candidate> candidates;
  Visitor output_visitor(context, sm, lang, candidates);
  output_visitor.TraverseStmt(const_cast<CompoundStmt *>(body));
  if (candidates.empty()) {
    return std::nullopt;
  }
  return std::max_element(candidates.begin(), candidates.end(),
                          [](const Candidate &left, const Candidate &right) {
                            return left.offset < right.offset;
                          })
      ->name;
}

class FactVisitor final : public RecursiveASTVisitor<FactVisitor> {
public:
  bool TraverseLambdaExpr(clang::LambdaExpr *) { return true; }

  bool VisitCallExpr(CallExpr *call) {
    const auto *callee = directCallee(call);
    if (callee == nullptr) {
      return true;
    }
    const std::string name = callee->getNameAsString();
    calls[name].push_back(call);
    return true;
  }

  bool VisitForStmt(ForStmt *loop) {
    const auto *init = llvm::dyn_cast_or_null<clang::DeclStmt>(loop->getInit());
    if (init == nullptr || !init->isSingleDecl()) {
      return true;
    }
    const auto *variable = llvm::dyn_cast<VarDecl>(init->getSingleDecl());
    const auto *condition =
        llvm::dyn_cast_or_null<BinaryOperator>(loop->getCond());
    if (variable == nullptr || !variable->hasInit() || condition == nullptr ||
        condition->getOpcode() != clang::BO_LT) {
      return true;
    }
    if (!containsName(condition->getLHS(), variable->getNameAsString()) ||
        (!containsLayerBound(condition->getRHS()) &&
         !containsName(condition->getRHS(), "il_end"))) {
      return true;
    }
    layer_loops.push_back(loop);
    return true;
  }

  bool VisitDeclRefExpr(DeclRefExpr *ref) {
    has_stage_filter |= ref->getDecl()->getNameAsString() == "stage_filter";
    return true;
  }

  std::map<std::string, std::vector<const CallExpr *>> calls;
  std::vector<const ForStmt *> layer_loops;
  bool has_stage_filter = false;
};

bool isCallbackOnly(const Stmt *statement) {
  class Visitor final : public RecursiveASTVisitor<Visitor> {
  public:
    bool TraverseLambdaExpr(clang::LambdaExpr *) {
      valid_ = false;
      return false;
    }

    bool VisitCallExpr(CallExpr *call) {
      const auto *callee = directCallee(call);
      if (callee == nullptr || callee->getNameAsString() != "cb") {
        valid_ = false;
        return false;
      }
      saw_callback_ = true;
      return true;
    }

    bool VisitBinaryOperator(BinaryOperator *binary) {
      if (binary->isAssignmentOp()) {
        valid_ = false;
        return false;
      }
      return true;
    }

    bool valid() const { return valid_ && saw_callback_; }

  private:
    bool valid_ = true;
    bool saw_callback_ = false;
  } visitor;
  visitor.TraverseStmt(const_cast<Stmt *>(statement));
  return visitor.valid();
}

std::vector<const BinaryOperator *> assignmentsTo(const Stmt *root,
                                                  llvm::StringRef variable) {
  class AssignmentVisitor final
      : public RecursiveASTVisitor<AssignmentVisitor> {
  public:
    AssignmentVisitor(llvm::StringRef variable,
                      std::vector<const BinaryOperator *> &results)
        : variable_(variable), results_(results) {}

    bool TraverseLambdaExpr(clang::LambdaExpr *) { return true; }

    bool VisitBinaryOperator(BinaryOperator *binary) {
      if (!binary->isAssignmentOp()) {
        return true;
      }
      const auto lhs = referencedName(binary->getLHS());
      if (lhs && *lhs == variable_) {
        results_.push_back(binary);
      }
      return true;
    }

  private:
    llvm::StringRef variable_;
    std::vector<const BinaryOperator *> &results_;
  };

  std::vector<const BinaryOperator *> assignments;
  AssignmentVisitor visitor(variable, assignments);
  visitor.TraverseStmt(const_cast<Stmt *>(root));
  return assignments;
}

std::optional<HyperconnectionPrelude> hyperconnectionPrelude(
    const CompoundStmt *constructor_body, const ForStmt *loop,
    llvm::StringRef embedding_activation, llvm::StringRef carried,
    const SourceManager &sm, const clang::LangOptions &lang) {
  if (embedding_activation == carried) {
    return std::nullopt;
  }
  const auto loop_offset = fileOffset(loop->getBeginLoc(), sm);
  if (!loop_offset) {
    return std::nullopt;
  }

  const VarDecl *carried_decl = nullptr;
  class DeclVisitor final : public RecursiveASTVisitor<DeclVisitor> {
  public:
    DeclVisitor(llvm::StringRef carried, uint64_t loop_offset,
                const SourceManager &sm, const VarDecl *&result)
        : carried_(carried), loop_offset_(loop_offset), sm_(sm), result_(result) {}

    bool TraverseLambdaExpr(clang::LambdaExpr *) { return true; }

    bool VisitVarDecl(VarDecl *decl) {
      const auto offset = fileOffset(decl->getBeginLoc(), sm_);
      if (decl->getName() == carried_ && decl->hasInit() && offset &&
          *offset < loop_offset_) {
        result_ = result_ == nullptr ? decl : nullptr;
        ambiguous_ |= result_ == nullptr;
      }
      return true;
    }

    bool ambiguous() const { return ambiguous_; }

  private:
    llvm::StringRef carried_;
    uint64_t loop_offset_;
    const SourceManager &sm_;
    const VarDecl *&result_;
    bool ambiguous_ = false;
  } decl_visitor(carried, *loop_offset, sm, carried_decl);
  decl_visitor.TraverseStmt(const_cast<CompoundStmt *>(constructor_body));
  if (decl_visitor.ambiguous() || carried_decl == nullptr) {
    return std::nullopt;
  }

  const BinaryOperator *repeat_assignment = nullptr;
  const CallExpr *repeat_call = nullptr;
  for (const BinaryOperator *assignment : assignmentsTo(constructor_body, carried)) {
    const auto offset = fileOffset(assignment->getBeginLoc(), sm);
    if (!offset || *offset >= *loop_offset) {
      continue;
    }
    const auto *call = llvm::dyn_cast<CallExpr>(
        assignment->getRHS()->IgnoreParenImpCasts());
    const auto *callee = call == nullptr ? nullptr : directCallee(call);
    if (callee == nullptr || callee->getNameAsString() != "ggml_repeat_4d" ||
        call->getNumArgs() != 6 ||
        sourceText(call->getArg(5)->getSourceRange(), sm, lang) != "1" ||
        !containsName(call->getArg(1), carried)) {
      return std::nullopt;
    }
    if (repeat_assignment != nullptr) {
      return std::nullopt;
    }
    repeat_assignment = assignment;
    repeat_call = call;
  }
  if (repeat_assignment == nullptr || repeat_call == nullptr) {
    return std::nullopt;
  }
  const Stmt *repeat_statement =
      directChildContaining(constructor_body, repeat_assignment, sm, lang);
  if (repeat_statement == nullptr) {
    return std::nullopt;
  }

  HyperconnectionPrelude result;
  result.carried_decl = carried_decl;
  result.repeat_assignment = repeat_assignment;
  result.repeat_statement = repeat_statement;
  result.width = sourceText(repeat_call->getArg(2)->getSourceRange(), sm, lang);
  result.multiplicity =
      sourceText(repeat_call->getArg(3)->getSourceRange(), sm, lang);
  result.tokens = sourceText(repeat_call->getArg(4)->getSourceRange(), sm, lang);
  if (result.width.empty() || result.multiplicity.empty() || result.tokens.empty()) {
    return std::nullopt;
  }
  return result;
}

bool addReplace(std::vector<Edit> &edits, llvm::StringRef kind,
                llvm::StringRef file, clang::SourceRange range,
                llvm::StringRef replacement, const SourceManager &sm,
                const clang::LangOptions &lang) {
  const auto bytes = tokenRange(range, sm, lang);
  if (!bytes) {
    return false;
  }
  edits.push_back(Edit{kind.str(), file.str(), bytes->first, bytes->second,
                       replacement.str()});
  return true;
}

bool addInsert(std::vector<Edit> &edits, llvm::StringRef kind,
               llvm::StringRef file, SourceLocation location,
               llvm::StringRef text, const SourceManager &sm) {
  const auto offset = fileOffset(location, sm);
  if (!offset) {
    return false;
  }
  edits.push_back(Edit{kind.str(), file.str(), *offset, 0, text.str()});
  return true;
}

bool nonOverlapping(std::vector<Edit> edits) {
  std::sort(edits.begin(), edits.end(),
            [](const Edit &left, const Edit &right) {
              return std::tie(left.file, left.offset, left.length) <
                     std::tie(right.file, right.offset, right.length);
            });
  for (size_t i = 1; i < edits.size(); ++i) {
    if (edits[i - 1].file != edits[i].file) {
      continue;
    }
    if (edits[i - 1].offset == edits[i].offset ||
        edits[i - 1].offset + edits[i - 1].length > edits[i].offset) {
      return false;
    }
  }
  return true;
}

class BuilderCallback final : public MatchFinder::MatchCallback {
public:
  void run(const MatchFinder::MatchResult &result) override {
    const auto *constructor =
        result.Nodes.getNodeAs<CXXConstructorDecl>("constructor");
    if (constructor == nullptr || constructor->getBody() == nullptr ||
        result.SourceManager == nullptr || result.Context == nullptr) {
      return;
    }
    if (constructor->getParent()->isLambda()) {
      return;
    }

    ASTContext &context = *result.Context;
    const SourceManager &sm = *result.SourceManager;
    const auto &lang = context.getLangOpts();
    const SourceLocation location =
        sm.getSpellingLoc(constructor->getLocation());
    if (!location.isValid() || !sm.isWrittenInMainFile(location)) {
      return;
    }

    llvm::SmallString<256> canonical_file;
    if (llvm::sys::fs::real_path(sm.getFilename(location), canonical_file)) {
      return;
    }
    const std::string file = canonical_file.str().str();
    const std::string models_root = SourceRoot + "/src/models/";
    if (!llvm::StringRef(file).starts_with(models_root)) {
      return;
    }
    const std::string qualified = constructor->getQualifiedNameAsString();
    if (qualified.find("::graph") == std::string::npos) {
      return;
    }

    BuilderReport report;
    report.file = llvm::StringRef(file).drop_front(SourceRoot.size() + 1).str();
    report.constructor = qualified;
    report.line = sm.getSpellingLineNumber(location);

    FactVisitor facts;
    facts.TraverseStmt(const_cast<Stmt *>(constructor->getBody()));

    const auto *constructor_body =
        llvm::dyn_cast<CompoundStmt>(constructor->getBody());

    const bool has_begin = facts.calls["begin_block"].size() == 1;
    // A transformed loop has one terminal end marker plus one marker on each
    // loop-level continue path. More than one end marker is therefore valid.
    const bool has_end = !facts.calls["end_block"].empty();
    if (facts.has_stage_filter && has_begin && has_end) {
      report.verdict = "already_transformed";
      reports_.push_back(std::move(report));
      return;
    }
    if (has_begin != has_end ||
        (!facts.has_stage_filter && (has_begin || has_end))) {
      refuse(report, "partial stage transformation");
      reports_.push_back(std::move(report));
      return;
    }

    // MTP/draft heads and encoder sidecars execute in a distinct context
    // attached to the final pipeline stage. They do not consume the primary
    // decoder's [layer_start, layer_end) interval and must never receive the
    // generic stage-loop rewrite. Prove that role from the builder type or its
    // graph inputs/results rather than from an architecture or file name.
    const bool typed_mtp_builder =
        llvm::StringRef(qualified).contains("::graph_mtp::graph_mtp");
    const bool context_sidecar =
        constructor_body != nullptr && containsName(constructor_body, "ctx_other");
    const bool encoder_sidecar =
        facts.calls["build_inp_embd_enc"].size() == 1 &&
        constructor_body != nullptr &&
        containsName(constructor_body, "t_h_nextn");
    if (!facts.has_stage_filter &&
        (typed_mtp_builder || context_sidecar || encoder_sidecar)) {
      report.verdict = "supported_auxiliary";
      report.proof.execution_scope = "final_stage_sidecar";
      if (typed_mtp_builder) {
        report.proof.scope_evidence.emplace_back("typed_mtp_builder");
      }
      if (context_sidecar) {
        report.proof.scope_evidence.emplace_back("cross_context_input");
      }
      if (encoder_sidecar) {
        report.proof.scope_evidence.emplace_back("encoder_sidecar_output");
      }
      reports_.push_back(std::move(report));
      return;
    }
    const bool completing_filter = facts.has_stage_filter;

    const auto &embedding_calls = facts.calls["build_inp_embd"];
    const CallExpr *embedding = nullptr;
    bool standard_embedding = false;
    if (embedding_calls.size() == 1 &&
        embedding_calls.front()->getNumArgs() > 0) {
      embedding = embedding_calls.front();
      standard_embedding = true;
    } else if (embedding_calls.empty()) {
      std::vector<const CallExpr *> token_gathers;
      for (const CallExpr *call : facts.calls["ggml_get_rows"]) {
        if (call->getNumArgs() >= 2 &&
            containsName(call->getArg(1), "tok_embd")) {
          token_gathers.push_back(call);
        }
      }
      if (token_gathers.size() == 1) {
        embedding = token_gathers.front();
      }
    }
    if (embedding == nullptr) {
      refuse(report, "cannot prove a unique token embedding producer");
      reports_.push_back(std::move(report));
      return;
    }
    const auto activation = assignedName(embedding, context);
    const Stmt *embedding_statement =
        constructor_body == nullptr
            ? nullptr
            : directChildContaining(constructor_body, embedding, sm, lang);
    if (!activation || embedding_statement == nullptr) {
      refuse(report, "cannot prove the embedding activation owner");
      reports_.push_back(std::move(report));
      return;
    }
    report.proof.embedding_owner = true;

    if (facts.layer_loops.empty()) {
      refuse(report, "no layer block loop");
      reports_.push_back(std::move(report));
      return;
    }
    std::vector<std::pair<int, const ForStmt *>> scored_loops;
    for (const ForStmt *candidate : facts.layer_loops) {
      scored_loops.emplace_back(
          layerLoopScore(candidate, *activation, sm, lang), candidate);
    }
    const int best_score =
        std::max_element(scored_loops.begin(), scored_loops.end(),
                         [](const auto &left, const auto &right) {
                           return left.first < right.first;
                         })
            ->first;
    std::vector<const ForStmt *> best_loops;
    for (const auto &[score, candidate] : scored_loops) {
      if (score == best_score) {
        best_loops.push_back(candidate);
      }
    }
    if (best_score <= 0 || best_loops.size() != 1) {
      if (best_score > 0 && best_loops.size() > 1) {
        std::vector<std::string> domains;
        for (const ForStmt *candidate : best_loops) {
          const auto *candidate_condition =
              llvm::dyn_cast_or_null<BinaryOperator>(candidate->getCond());
          if (candidate_condition == nullptr) {
            domains.clear();
            break;
          }
          domains.push_back(sourceText(
              candidate_condition->getRHS()->getSourceRange(), sm, lang));
        }
        std::sort(domains.begin(), domains.end());
        domains.erase(std::unique(domains.begin(), domains.end()), domains.end());
        if (domains.size() > 1) {
          report.verdict = "supported_whole_model";
          report.proof.execution_scope = "multiple_sequential_layer_domains";
          report.proof.scope_evidence = std::move(domains);
          reports_.push_back(std::move(report));
          return;
        }
      }
      refuse(report, best_loops.size() == 1
                         ? "cannot prove selected layer block loop"
                         : "multiple equally ranked layer block loops");
      reports_.push_back(std::move(report));
      return;
    }
    const ForStmt *loop = best_loops.front();
    const auto *loop_body = llvm::dyn_cast<CompoundStmt>(loop->getBody());
    const auto *init = llvm::cast<clang::DeclStmt>(loop->getInit());
    const auto *loop_var = llvm::cast<VarDecl>(init->getSingleDecl());
    const auto *condition = llvm::cast<BinaryOperator>(loop->getCond());
    report.proof.loop_var = loop_var->getNameAsString();
    report.proof.loop_start =
        sourceText(loop_var->getInit()->getSourceRange(), sm, lang);
    report.proof.loop_end = stableLoopEnd(condition->getRHS(), sm, lang);
    const std::string expected_loop_start =
        completing_filter ? "il_start" : "0";
    if (loop_body == nullptr ||
        report.proof.loop_start != expected_loop_start) {
      refuse(report,
             loop_body == nullptr
                 ? "block loop body is not compound"
                 : "block loop start does not match transformation state");
      reports_.push_back(std::move(report));
      return;
    }

    ExitVisitor exits(context, loop);
    exits.TraverseStmt(const_cast<CompoundStmt *>(loop_body));
    report.proof.nonlocal_exits = exits.exits;
    if (!exits.exits.empty()) {
      refuse(report, "block loop contains a non-local exit");
      reports_.push_back(std::move(report));
      return;
    }

    const auto carried =
        layerCarriedName(loop_body, *activation, context, sm, lang);
    if (!carried) {
      refuse(report, "cannot prove the layer-carried activation");
      reports_.push_back(std::move(report));
      return;
    }
    report.proof.activation_in = *carried;
    report.proof.activation_out = *carried;

    const auto hyperconnection =
        hyperconnectionPrelude(constructor_body, loop, *activation, *carried,
                               sm, lang);
    if (!completing_filter && *activation != *carried && !hyperconnection) {
      refuse(report,
             "layer-carried activation differs from the embedding without a "
             "proven hyperconnection prelude");
      reports_.push_back(std::move(report));
      return;
    }
    if (hyperconnection) {
      report.proof.scope_evidence.emplace_back(
          "hyperconnection_activation_frontier");
    }

    std::vector<const BinaryOperator *> preloop_activation_assignments;
    if (!completing_filter && !hyperconnection && *activation == *carried) {
      const auto embedding_end = tokenRange(embedding_statement->getSourceRange(), sm, lang);
      const auto loop_begin = fileOffset(loop->getBeginLoc(), sm);
      if (!embedding_end || !loop_begin) {
        refuse(report, "cannot locate the pre-loop activation region");
        reports_.push_back(std::move(report));
        return;
      }
      const uint64_t prelude_begin = embedding_end->first + embedding_end->second;
      for (const BinaryOperator *assignment :
           assignmentsTo(constructor_body, *carried)) {
        const auto offset = fileOffset(assignment->getBeginLoc(), sm);
        if (offset && *offset >= prelude_begin && *offset < *loop_begin) {
          preloop_activation_assignments.push_back(assignment);
        }
      }
      if (!preloop_activation_assignments.empty()) {
        report.proof.scope_evidence.emplace_back(
            "guarded_embedding_prelude");
      }
    }

    const auto &output_calls = facts.calls["build_inp_out_ids"];
    if (output_calls.size() > 1 && !completing_filter) {
      refuse(report, "multiple build_inp_out_ids calls");
      reports_.push_back(std::move(report));
      return;
    }
    const CallExpr *output_call = output_calls.size() == 1
                                      ? output_calls.front()
                                      : nullptr;
    report.proof.output_owner = !output_calls.empty();

    std::vector<const IfStmt *> terminal_ifs;
    class TerminalVisitor final : public RecursiveASTVisitor<TerminalVisitor> {
    public:
      TerminalVisitor(llvm::StringRef loop_var,
                      std::vector<const IfStmt *> &results)
          : loop_var_(loop_var), results_(results) {}

      bool VisitIfStmt(IfStmt *statement) {
        const Expr *condition = statement->getCond();
        if (containsName(condition, loop_var_) &&
            containsName(condition, "n_layer") &&
            containsName(condition, "inp_out_ids")) {
          results_.push_back(statement);
        }
        return true;
      }

    private:
      llvm::StringRef loop_var_;
      std::vector<const IfStmt *> &results_;
    } terminal_visitor(report.proof.loop_var, terminal_ifs);
    terminal_visitor.TraverseStmt(const_cast<CompoundStmt *>(loop_body));
    for (const IfStmt *terminal : terminal_ifs) {
      report.proof.terminal_predicates.push_back(
          sourceText(terminal->getCond()->getSourceRange(), sm, lang));
    }

    const std::string indent =
        indentationAt(embedding_statement->getBeginLoc(), sm);
    const std::string inner_indent =
        indentationAt((*loop_body->body_begin())->getBeginLoc(), sm);
    const std::string original_embedding =
        sourceText(embedding->getSourceRange(), sm, lang);
    const std::string declarations =
        "const skippy_graph_filter & stage_filter = build_inputs.filter;\n" +
        indent + "const bool stage_filtered = stage_filter.enabled;\n" +
        indent +
        "const int il_start = stage_filtered ? stage_filter.layer_start : "
        "0;\n" +
        indent +
        "const int il_end   = stage_filtered ? stage_filter.layer_end   : " +
        report.proof.loop_end + ";\n\n" + indent;

    bool valid = true;
    if (!completing_filter) {
      valid &=
          addInsert(report.edits, "insert_filter_declarations", report.file,
                    embedding_statement->getBeginLoc(), declarations, sm);
      if (hyperconnection) {
        valid &= addReplace(
            report.edits, "rewrite_hyperconnection_embedding_owner",
            report.file, embedding->getSourceRange(),
            "(!stage_filtered || il_start == 0) ? " + original_embedding +
                " : nullptr",
            sm, lang);

        const Expr *carried_initializer = hyperconnection->carried_decl->getInit();
        const std::string original_initializer =
            sourceText(carried_initializer->getSourceRange(), sm, lang);
        valid &= addReplace(
            report.edits, "rewrite_hyperconnection_initializer", report.file,
            carried_initializer->getSourceRange(),
            "stage_filtered && il_start > 0 ? nullptr : " +
                original_initializer,
            sm, lang);

        const std::string repeat_indent =
            indentationAt(hyperconnection->repeat_statement->getBeginLoc(), sm);
        const std::string import =
            "if (stage_filtered && il_start > 0) {\n" + repeat_indent +
            "    auto stage_inp = "
            "std::make_unique<llm_graph_input_hyperconnection>(" +
            hyperconnection->width + ", " + hyperconnection->multiplicity +
            ");\n" + repeat_indent +
            "    stage_inp->values = ggml_new_tensor_3d(ctx0, GGML_TYPE_F32, " +
            hyperconnection->width + ", " + hyperconnection->multiplicity +
            ", " + hyperconnection->tokens + ");\n" + repeat_indent +
            "    cb(stage_inp->values, \"hc_stage_input\", -1);\n" +
            repeat_indent + "    ggml_set_input(stage_inp->values);\n" +
            repeat_indent + "    " + *carried + " = stage_inp->values;\n" +
            repeat_indent + "    res->t_skippy_activation_input = " + *carried +
            ";\n" + repeat_indent +
            "    res->add_input(std::move(stage_inp));\n" + repeat_indent +
            "}\n" + repeat_indent;
        valid &= addInsert(report.edits, "insert_hyperconnection_import",
                           report.file,
                           hyperconnection->repeat_statement->getBeginLoc(),
                           import, sm);

        const Expr *repeat_rhs = hyperconnection->repeat_assignment->getRHS();
        const std::string original_repeat =
            sourceText(repeat_rhs->getSourceRange(), sm, lang);
        valid &= addReplace(
            report.edits, "guard_hyperconnection_repeat", report.file,
            repeat_rhs->getSourceRange(),
            "stage_filtered && il_start > 0 ? " + *carried + " : " +
                original_repeat,
            sm, lang);
      } else if (standard_embedding) {
        const std::string original_argument =
            sourceText(embedding->getArg(0)->getSourceRange(), sm, lang);
        valid &= addReplace(report.edits, "rewrite_embedding_owner",
                            report.file,
                            embedding->getArg(0)->getSourceRange(),
                            "stage_filtered && il_start > 0 ? nullptr : " +
                                original_argument,
                            sm, lang);
      } else {
        valid &= addReplace(
            report.edits, "rewrite_manual_embedding_owner", report.file,
            embedding->getSourceRange(),
            "stage_filtered && il_start > 0 ? build_inp_embd(nullptr) : " +
                original_embedding,
            sm, lang);
      }

      std::set<const IfStmt *> guarded_ifs;
      for (const BinaryOperator *assignment : preloop_activation_assignments) {
        const Stmt *statement = directChildContaining(
            constructor_body, assignment, sm, lang);
        if (const auto *conditional = llvm::dyn_cast_or_null<IfStmt>(statement)) {
          if (guarded_ifs.insert(conditional).second) {
            const Expr *condition = conditional->getCond();
            valid &= addReplace(
                report.edits, "guard_embedding_prelude", report.file,
                condition->getSourceRange(),
                "(!stage_filtered || il_start == 0) && (" +
                    sourceText(condition->getSourceRange(), sm, lang) + ")",
                sm, lang);
          }
          continue;
        }
        if (statement != assignment) {
          valid = false;
          continue;
        }
        const Expr *rhs = assignment->getRHS();
        valid &= addReplace(
            report.edits, "guard_embedding_prelude", report.file,
            rhs->getSourceRange(),
            "stage_filtered && il_start > 0 ? " + *carried + " : " +
                sourceText(rhs->getSourceRange(), sm, lang),
            sm, lang);
      }
      valid &= addReplace(report.edits, "rewrite_loop_start", report.file,
                          loop_var->getInit()->getSourceRange(), "il_start", sm,
                          lang);
      valid &=
          addReplace(report.edits, "rewrite_loop_end", report.file,
                     condition->getRHS()->getSourceRange(), "il_end", sm, lang);
    }

    const auto body_begin = clang::Lexer::getLocForEndOfToken(
        loop_body->getLBracLoc(), 0, sm, lang);
    valid &=
        addInsert(report.edits, "insert_begin_block", report.file, body_begin,
                  "\n" + inner_indent + "begin_block(" + *carried + ", " +
                      report.proof.loop_var + ");\n",
                  sm);
    const std::string loop_indent =
        indentationAt(loop_body->getRBracLoc(), sm);
    const std::string end_block_indent =
        llvm::StringRef(inner_indent).starts_with(loop_indent)
            ? llvm::StringRef(inner_indent).drop_front(loop_indent.size()).str()
            : inner_indent;
    valid &= addInsert(report.edits, "insert_end_block", report.file,
                       loop_body->getRBracLoc(),
                       end_block_indent + "end_block(" + *carried + ", " +
                           report.proof.loop_var + ");\n" + loop_indent,
                       sm);
    for (const clang::ContinueStmt *statement : exits.continues) {
      const std::string continue_indent =
          indentationAt(statement->getBeginLoc(), sm);
      valid &= addInsert(report.edits, "insert_end_block_before_continue",
                         report.file, statement->getBeginLoc(),
                         "end_block(" + *carried + ", " +
                             report.proof.loop_var + ");\n" + continue_indent,
                         sm);
    }

    if (!completing_filter && output_call != nullptr) {
      const Stmt *output_statement =
          directChildContaining(constructor_body, output_call, sm, lang);
      if (output_statement == nullptr) {
        valid = false;
      } else {
        const auto assigned_output = assignedName(output_call, context);
        if (!assigned_output) {
          valid = false;
        } else {
          valid &= addReplace(
              report.edits, "rewrite_output_owner", report.file,
              output_call->getSourceRange(),
              "(!stage_filtered || stage_filter.include_output) ? " +
                  sourceText(output_call->getSourceRange(), sm, lang) +
                  " : nullptr",
              sm, lang);
        }
      }
    }

    for (const IfStmt *terminal : terminal_ifs) {
      class NLayerVisitor final : public RecursiveASTVisitor<NLayerVisitor> {
      public:
        explicit NLayerVisitor(std::vector<const Expr *> &matches)
            : matches_(matches) {}

        bool VisitDeclRefExpr(DeclRefExpr *ref) {
          if (ref->getDecl()->getNameAsString() == "n_layer") {
            matches_.push_back(ref);
          }
          return true;
        }

        bool VisitMemberExpr(clang::MemberExpr *member) {
          if (member->getMemberDecl()->getNameAsString() == "n_layer") {
            matches_.push_back(member);
          }
          return true;
        }

      private:
        std::vector<const Expr *> &matches_;
      } n_layer_visitor(n_layer_refs_);
      n_layer_refs_.clear();
      n_layer_visitor.TraverseStmt(const_cast<Expr *>(terminal->getCond()));
      for (const Expr *ref : n_layer_refs_) {
        if (sourceText(ref->getSourceRange(), sm, lang) != "n_layer") {
          continue;
        }
        valid &=
            addReplace(report.edits, "rewrite_terminal_endpoint", report.file,
                       ref->getSourceRange(), "il_end", sm, lang);
      }
    }

    if (!completing_filter) {
      const SourceLocation after_loop =
          clang::Lexer::getLocForEndOfToken(loop->getEndLoc(), 0, sm, lang);
      const std::string boundary =
          "\n" + indent +
          "if (stage_filtered && !stage_filter.include_output) {\n" + indent +
          "    cb(" + *carried + ", \"stage_boundary\", il_end - 1);\n" +
          indent + "    res->t_embd = " + *carried + ";\n" + indent +
          "    ggml_build_forward_expand(gf, " + *carried + ");\n" + indent +
          "    return;\n" + indent + "}\n";
      valid &= addInsert(report.edits, "insert_stage_boundary", report.file,
                         after_loop, boundary, sm);
    }

    if (!valid || !nonOverlapping(report.edits)) {
      report.edits.clear();
      refuse(report, valid ? "planned edits overlap"
                           : "cannot map edit to source bytes");
    } else {
      report.verdict = "transformable";
    }
    reports_.push_back(std::move(report));
  }

  int finish() {
    std::sort(reports_.begin(), reports_.end(),
              [](const auto &left, const auto &right) {
                return std::tie(left.file, left.line, left.constructor) <
                       std::tie(right.file, right.line, right.constructor);
              });

    int apply_result = 0;
    if (Apply) {
      apply_result = applyEdits();
    }
    const int report_result = writeReport();
    return apply_result != 0 ? apply_result : report_result;
  }

private:
  static void refuse(BuilderReport &report, llvm::StringRef reason) {
    report.verdict = "unsupported_shape";
    report.unsupported_reason = reason.str();
    report.edits.clear();
  }

  int applyEdits() {
    std::map<std::string, std::vector<Edit>> by_file;
    for (const auto &report : reports_) {
      if (report.verdict != "transformable") {
        continue;
      }
      for (const auto &edit : report.edits) {
        by_file[edit.file].push_back(edit);
      }
    }
    for (auto &[file, edits] : by_file) {
      const std::string source_file = SourceRoot + "/" + file;
      std::ifstream input(source_file, std::ios::binary);
      if (!input) {
        llvm::errs() << "cannot read " << source_file << "\n";
        return 1;
      }
      std::string contents((std::istreambuf_iterator<char>(input)),
                           std::istreambuf_iterator<char>());
      std::sort(edits.begin(), edits.end(),
                [](const Edit &left, const Edit &right) {
                  return std::tie(left.offset, left.length) >
                         std::tie(right.offset, right.length);
                });
      for (const auto &edit : edits) {
        if (edit.offset + edit.length > contents.size()) {
          llvm::errs() << "edit outside file " << source_file << "\n";
          return 1;
        }
        contents.replace(edit.offset, edit.length, edit.text);
      }
      std::error_code error;
      llvm::raw_fd_ostream output(source_file, error, llvm::sys::fs::OF_None);
      if (error) {
        llvm::errs() << "cannot write " << source_file << ": "
                     << error.message() << "\n";
        return 1;
      }
      output << contents;
    }
    return 0;
  }

  int writeReport() const {
    llvm::json::Array builders;
    int64_t transformable = 0;
    int64_t already_transformed = 0;
    int64_t unsupported_shape = 0;
    int64_t supported_auxiliary = 0;
    int64_t supported_whole_model = 0;
    int64_t errors = 0;
    for (const auto &report : reports_) {
      if (report.verdict == "transformable") {
        ++transformable;
      } else if (report.verdict == "already_transformed") {
        ++already_transformed;
      } else if (report.verdict == "unsupported_shape") {
        ++unsupported_shape;
      } else if (report.verdict == "supported_auxiliary") {
        ++supported_auxiliary;
      } else if (report.verdict == "supported_whole_model") {
        ++supported_whole_model;
      } else {
        ++errors;
      }

      llvm::json::Array edits;
      for (const auto &edit : report.edits) {
        edits.push_back(llvm::json::Object{
            {"file", edit.file},
            {"kind", edit.kind},
            {"range", llvm::json::Array{static_cast<int64_t>(edit.offset),
                                        static_cast<int64_t>(edit.offset +
                                                             edit.length)}},
            {"text", edit.text},
        });
      }
      llvm::json::Array predicates;
      for (const auto &predicate : report.proof.terminal_predicates) {
        predicates.push_back(predicate);
      }
      llvm::json::Array exits;
      for (const auto &exit : report.proof.nonlocal_exits) {
        exits.push_back(exit);
      }
      llvm::json::Object proof{
          {"activation_in", report.proof.activation_in},
          {"activation_out", report.proof.activation_out},
          {"embedding_owner", report.proof.embedding_owner},
          {"loop", llvm::json::Object{{"end", report.proof.loop_end},
                                      {"start", report.proof.loop_start},
                                      {"var", report.proof.loop_var}}},
          {"nonlocal_exits", std::move(exits)},
          {"execution_scope", report.proof.execution_scope},
          {"scope_evidence", [&report]() {
             llvm::json::Array evidence;
             for (const auto &item : report.proof.scope_evidence) {
               evidence.push_back(item);
             }
             return evidence;
           }()},
          {"output_owner", report.proof.output_owner},
          {"terminal_predicates", std::move(predicates)},
      };
      builders.push_back(llvm::json::Object{
          {"constructor", report.constructor},
          {"edits", std::move(edits)},
          {"file", report.file},
          {"line", static_cast<int64_t>(report.line)},
          {"proof", std::move(proof)},
          {"unsupported_reason",
           report.unsupported_reason.empty()
               ? llvm::json::Value(nullptr)
               : llvm::json::Value(report.unsupported_reason)},
          {"verdict", report.verdict},
      });
    }

    std::error_code error;
    llvm::raw_fd_ostream output(ReportPath, error);
    if (error) {
      llvm::errs() << "cannot write report " << ReportPath << ": "
                   << error.message() << "\n";
      return 1;
    }
    output << llvm::formatv(
        "{0:2}\n",
        llvm::json::Value(llvm::json::Object{
            {"builders", std::move(builders)},
            {"generator_version", "0.2.0"},
            {"llama_cpp_commit", LlamaCommit},
            {"schema_version", 1},
            {"source_root", SourceRoot},
            {"summary",
             llvm::json::Object{{"already_transformed", already_transformed},
                                {"error", errors},
                                {"supported_auxiliary", supported_auxiliary},
                                {"supported_whole_model", supported_whole_model},
                                {"transformable", transformable},
                                {"unsupported_shape", unsupported_shape}}},
        }));
    return 0;
  }

  std::vector<const Expr *> n_layer_refs_;
  std::vector<BuilderReport> reports_;
};

} // namespace

int main(int argc, const char **argv) {
  auto parser = clang::tooling::CommonOptionsParser::create(
      argc, argv, RewriterCategory, llvm::cl::OneOrMore);
  if (!parser) {
    llvm::errs() << llvm::toString(parser.takeError());
    return 1;
  }

  llvm::SmallString<256> canonical_source_root;
  if (const std::error_code error =
          llvm::sys::fs::real_path(SourceRoot, canonical_source_root)) {
    llvm::errs() << "cannot resolve source root " << SourceRoot << ": "
                 << error.message() << "\n";
    return 1;
  }
  SourceRoot = canonical_source_root.str().str();

  clang::tooling::ClangTool tool(parser->getCompilations(),
                                 parser->getSourcePathList());
  BuilderCallback callback;
  MatchFinder finder;
  finder.addMatcher(
      cxxConstructorDecl(isDefinition(), isExpansionInMainFile(),
                         unless(clang::ast_matchers::isTemplateInstantiation()))
          .bind("constructor"),
      &callback);

  const int run_result =
      tool.run(clang::tooling::newFrontendActionFactory(&finder).get());
  if (run_result != 0) {
    return run_result;
  }
  return callback.finish();
}
