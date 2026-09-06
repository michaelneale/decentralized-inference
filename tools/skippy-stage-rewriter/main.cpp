#include "clang/AST/ASTContext.h"
#include "clang/AST/DeclCXX.h"
#include "clang/AST/Expr.h"
#include "clang/AST/ParentMapContext.h"
#include "clang/AST/RecursiveASTVisitor.h"
#include "clang/AST/Stmt.h"
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

bool containsName(const Expr *expr, llvm::StringRef target) {
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
  visitor.TraverseStmt(const_cast<Expr *>(expr));
  return visitor.found();
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

  bool VisitBreakStmt(clang::BreakStmt *) {
    exits.emplace_back("break");
    return true;
  }

  bool VisitContinueStmt(clang::ContinueStmt *) {
    exits.emplace_back("continue");
    return true;
  }

  std::vector<std::string> exits;
};

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
        (!containsName(condition->getRHS(), "n_layer") &&
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

    const bool has_begin = facts.calls["begin_block"].size() == 1;
    const bool has_end = facts.calls["end_block"].size() == 1;
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
    const bool completing_filter = facts.has_stage_filter;

    if (facts.layer_loops.size() != 1) {
      refuse(report, facts.layer_loops.empty()
                         ? "no unique n_layer block loop"
                         : "multiple n_layer block loops");
      reports_.push_back(std::move(report));
      return;
    }
    const ForStmt *loop = facts.layer_loops.front();
    const auto *loop_body = llvm::dyn_cast<CompoundStmt>(loop->getBody());
    const auto *init = llvm::cast<clang::DeclStmt>(loop->getInit());
    const auto *loop_var = llvm::cast<VarDecl>(init->getSingleDecl());
    const auto *condition = llvm::cast<BinaryOperator>(loop->getCond());
    report.proof.loop_var = loop_var->getNameAsString();
    report.proof.loop_start =
        sourceText(loop_var->getInit()->getSourceRange(), sm, lang);
    report.proof.loop_end =
        sourceText(condition->getRHS()->getSourceRange(), sm, lang);
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

    ExitVisitor exits;
    exits.TraverseStmt(const_cast<CompoundStmt *>(loop_body));
    report.proof.nonlocal_exits = exits.exits;
    if (!exits.exits.empty()) {
      refuse(report, "block loop contains a non-local exit");
      reports_.push_back(std::move(report));
      return;
    }

    const auto &embedding_calls = facts.calls["build_inp_embd"];
    if (embedding_calls.size() != 1 ||
        embedding_calls.front()->getNumArgs() != 1) {
      refuse(report,
             "expected exactly one single-argument build_inp_embd call");
      reports_.push_back(std::move(report));
      return;
    }
    const CallExpr *embedding = embedding_calls.front();
    const auto activation = assignedName(embedding, context);
    const auto *constructor_body =
        llvm::dyn_cast<CompoundStmt>(constructor->getBody());
    const Stmt *embedding_statement =
        constructor_body == nullptr
            ? nullptr
            : directChildContaining(constructor_body, embedding, sm, lang);
    if (!activation || embedding_statement == nullptr) {
      refuse(report, "cannot prove the embedding activation owner");
      reports_.push_back(std::move(report));
      return;
    }
    report.proof.activation_in = *activation;
    report.proof.embedding_owner = true;

    const auto carries = assignmentsTo(loop_body, *activation);
    if (carries.size() != 1) {
      refuse(report, carries.empty()
                         ? "no unique carried activation assignment"
                         : "multiple carried activation assignments");
      reports_.push_back(std::move(report));
      return;
    }
    const BinaryOperator *carry = carries.front();
    const Stmt *carry_statement =
        directChildContaining(loop_body, carry, sm, lang);
    if (carry_statement == nullptr || loop_body->body_empty() ||
        carry_statement != loop_body->body_back()) {
      refuse(report,
             "carried activation assignment is not the final block statement");
      reports_.push_back(std::move(report));
      return;
    }
    const auto output = referencedName(carry->getRHS());
    if (!output) {
      refuse(report, "carried activation output is not a named tensor");
      reports_.push_back(std::move(report));
      return;
    }
    report.proof.activation_out = *output;

    const auto &output_calls = facts.calls["build_inp_out_ids"];
    if (output_calls.size() > 1) {
      refuse(report, "multiple build_inp_out_ids calls");
      reports_.push_back(std::move(report));
      return;
    }
    const CallExpr *output_call =
        output_calls.empty() ? nullptr : output_calls.front();
    report.proof.output_owner = output_call != nullptr;

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
    const std::string inner_indent = indentationAt(carry->getBeginLoc(), sm);
    const std::string original_embedding =
        sourceText(embedding->getArg(0)->getSourceRange(), sm, lang);
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
      valid &= addReplace(report.edits, "rewrite_embedding_owner", report.file,
                          embedding->getArg(0)->getSourceRange(),
                          "stage_filtered && il_start > 0 ? nullptr : " +
                              original_embedding,
                          sm, lang);
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
                  "\n" + inner_indent + "begin_block(" + *activation + ", " +
                      report.proof.loop_var + ");\n",
                  sm);
    valid &= addInsert(report.edits, "insert_end_block", report.file,
                       carry->getBeginLoc(),
                       "end_block(" + *output + ", " + report.proof.loop_var +
                           ");\n\n" + inner_indent,
                       sm);

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
          "    cb(" + *activation + ", \"stage_boundary\", il_end - 1);\n" +
          indent + "    res->t_embd = " + *activation + ";\n" + indent +
          "    ggml_build_forward_expand(gf, " + *activation + ");\n" + indent +
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
    int64_t errors = 0;
    for (const auto &report : reports_) {
      if (report.verdict == "transformable") {
        ++transformable;
      } else if (report.verdict == "already_transformed") {
        ++already_transformed;
      } else if (report.verdict == "unsupported_shape") {
        ++unsupported_shape;
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
            {"generator_version", "0.1.0"},
            {"llama_cpp_commit", LlamaCommit},
            {"schema_version", 0},
            {"source_root", SourceRoot},
            {"summary",
             llvm::json::Object{{"already_transformed", already_transformed},
                                {"error", errors},
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
