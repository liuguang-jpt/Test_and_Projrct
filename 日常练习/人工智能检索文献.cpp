// knowledge_retrieval.h
#ifndef KNOWLEDGE_RETRIEVAL_H
#define KNOWLEDGE_RETRIEVAL_H

#include <vector>
#include <string>
#include <unordered_map>
#include <map>
#include <memory>
#include <cmath>

// 命名空间
namespace KnowledgeRetrieval {

// 读者知识状态
struct ReaderKnowledge {
    // 基础信息
    std::string userId;
    std::vector<std::string> familiarTopics;  // 熟悉话题
    
    // 概念掌握度 (概念名 -> 掌握度 0.0-1.0)
    std::unordered_map<std::string, float> conceptMastery;
    
    // 知识缺口
    std::vector<std::string> knowledgeGaps;
    
    // 阅读水平
    enum class ReadingLevel { BEGINNER, INTERMEDIATE, ADVANCED, EXPERT };
    ReadingLevel readingLevel;
    
    // 学习目标
    enum class LearningGoal { OVERVIEW, DEEP_DIVE, METHOD_LEARNING, PROBLEM_SOLVING };
    LearningGoal goal;
    
    // 时间限制 (小时)
    float timeLimit;
    
    // 偏好设置
    std::vector<std::string> preferredFormats;  // pdf, html, video等
    int maxPapers;  // 期望获取的最大文献数
    
    ReaderKnowledge() 
        : readingLevel(ReadingLevel::INTERMEDIATE)
        , goal(LearningGoal::OVERVIEW)
        , timeLimit(5.0f)
        , maxPapers(20) {}
};

// 文献元数据
struct PaperMetadata {
    std::string id;
    std::string title;
    std::string abstract;
    std::vector<std::string> authors;
    int year;
    std::string venue;  // 会议/期刊
    
    // 概念关键词及其权重
    std::unordered_map<std::string, float> concepts;
    
    // 难度等级
    enum class Difficulty { 
        INTRODUCTORY = 1,    // 入门级
        INTERMEDIATE = 2,    // 中级
        ADVANCED = 3,        // 高级
        EXPERT = 4           // 专家级
    };
    Difficulty difficulty;
    
    // 学习类型
    enum class PaperType { 
        THEORETICAL,     // 理论
        EMPIRICAL,       // 实证
        REVIEW,          // 综述
        METHODOLOGICAL,  // 方法论
        TUTORIAL         // 教程
    };
    PaperType paperType;
    
    // 预估阅读时间 (小时)
    float estimatedReadingTime;
    
    // 引用数
    int citationCount;
    
    // 评分信息
    float relevanceScore;
    float knowledgeGapScore;
    float difficultyMatchScore;
    
    PaperMetadata()
        : year(0)
        , difficulty(Difficulty::INTERMEDIATE)
        , paperType(PaperType::THEORETICAL)
        , estimatedReadingTime(2.0f)
        , citationCount(0)
        , relevanceScore(0.0f)
        , knowledgeGapScore(0.0f)
        , difficultyMatchScore(0.0f) {}
};

// 检索结果
struct RetrievalResult {
    PaperMetadata paper;
    float finalScore;
    std::vector<std::string> reasons;  // 推荐理由
    std::vector<std::string> prerequisites;  // 建议先阅读的文献ID
    
    RetrievalResult() : finalScore(0.0f) {}
    
    // 排序比较函数
    bool operator<(const RetrievalResult& other) const {
        return finalScore > other.finalScore;  // 降序排列
    }
};

// 概念图谱节点
struct ConceptNode {
    std::string concept;
    std::vector<std::string> prerequisites;  // 前置概念
    std::vector<std::string> relatedConcepts;  // 相关概念
    int depth;  // 概念深度 (基础概念为1)
    float importance;  // 重要性权重
    
    ConceptNode() : depth(1), importance(1.0f) {}
};

// 知识图谱
class KnowledgeGraph {
private:
    std::unordered_map<std::string, ConceptNode> graph;
    
public:
    void addConcept(const std::string& concept, const ConceptNode& node) {
        graph[concept] = node;
    }
    
    bool hasConcept(const std::string& concept) const {
        return graph.find(concept) != graph.end();
    }
    
    const ConceptNode& getConcept(const std::string& concept) const {
        static ConceptNode emptyNode;
        auto it = graph.find(concept);
        if (it != graph.end()) {
            return it->second;
        }
        return emptyNode;
    }
    
    // 计算概念间的相关度
    float calculateConceptRelevance(const std::string& concept1, 
                                    const std::string& concept2) const {
        if (!hasConcept(concept1) || !hasConcept(concept2)) {
            return 0.0f;
        }
        
        // 简单实现：检查是否为前置关系
        const auto& node1 = getConcept(concept1);
        const auto& node2 = getConcept(concept2);
        
        // 如果concept1是concept2的前置概念
        for (const auto& pre : node2.prerequisites) {
            if (pre == concept1) return 0.8f;
        }
        
        // 如果concept2是concept1的前置概念
        for (const auto& pre : node1.prerequisites) {
            if (pre == concept2) return 0.8f;
        }
        
        // 检查是否有共同的相关概念
        int commonCount = 0;
        for (const auto& rel1 : node1.relatedConcepts) {
            for (const auto& rel2 : node2.relatedConcepts) {
                if (rel1 == rel2) {
                    commonCount++;
                }
            }
        }
        
        if (commonCount > 0) {
            return 0.3f + 0.1f * std::min(commonCount, 3);
        }
        
        return 0.0f;
    }
};

} // namespace KnowledgeRetrieval

#endif // KNOWLEDGE_RETRIEVAL_H
// retrieval_engine.h
#ifndef RETRIEVAL_ENGINE_H
#define RETRIEVAL_ENGINE_H

#include "knowledge_retrieval.h"
#include <algorithm>
#include <queue>
#include <functional>

namespace KnowledgeRetrieval {

// 检索配置
struct RetrievalConfig {
    // 权重配置
    struct Weights {
        float relevance = 0.25f;
        float knowledgeGap = 0.30f;
        float difficultyMatch = 0.20f;
        float timeEfficiency = 0.10f;
        float novelty = 0.10f;
        float goalMatch = 0.05f;
    } weights;
    
    // 阈值配置
    struct Thresholds {
        float minDifficultyMatch = 0.3f;    // 最低难度匹配度
        float maxPrerequisiteGap = 0.5f;    // 最大前置知识缺口
        float minRelevance = 0.1f;          // 最低相关性
    } thresholds;
    
    // 排序策略
    enum class SortStrategy { 
        SCORE_BASED,      // 纯分数排序
        LEARNING_PATH,    // 学习路径排序
        HYBRID            // 混合策略
    };
    SortStrategy strategy = SortStrategy::HYBRID;
    
    // 多样性控制
    int maxPapersPerTopic = 3;  // 每个话题最大文献数
};

// 智能检索引擎
class IntelligentRetrievalEngine {
private:
    // 文献数据库
    std::vector<PaperMetadata> paperDatabase;
    
    // 知识图谱
    KnowledgeGraph conceptGraph;
    
    // 配置
    RetrievalConfig config;
    
public:
    IntelligentRetrievalEngine(const RetrievalConfig& cfg = RetrievalConfig())
        : config(cfg) {}
    
    // 添加文献到数据库
    void addPaper(const PaperMetadata& paper) {
        paperDatabase.push_back(paper);
    }
    
    // 设置知识图谱
    void setKnowledgeGraph(const KnowledgeGraph& graph) {
        conceptGraph = graph;
    }
    
    // 主检索函数
    std::vector<RetrievalResult> retrievePapers(
        const std::string& query,
        const ReaderKnowledge& readerKnowledge) {
        
        std::vector<RetrievalResult> results;
        
        // 第一阶段：基础检索
        std::vector<PaperMetadata> initialResults = 
            performInitialRetrieval(query, readerKnowledge);
        
        // 第二阶段：评分和排序
        for (auto& paper : initialResults) {
            RetrievalResult result;
            result.paper = paper;
            
            // 计算各项分数
            result.finalScore = calculatePaperScore(paper, query, readerKnowledge);
            
            // 生成推荐理由
            result.reasons = generateRecommendationReasons(paper, readerKnowledge);
            
            // 获取前置文献建议
            result.prerequisites = suggestPrerequisites(paper, readerKnowledge);
            
            // 检查是否满足最低要求
            if (shouldIncludePaper(result, readerKnowledge)) {
                results.push_back(result);
            }
        }
        
        // 第三阶段：最终排序
        return finalRanking(results, readerKnowledge);
    }
    
private:
    // 初始检索
    std::vector<PaperMetadata> performInitialRetrieval(
        const std::string& query,
        const ReaderKnowledge& readerKnowledge) {
        
        std::vector<PaperMetadata> results;
        
        for (const auto& paper : paperDatabase) {
            // 计算基础相关性
            float relevance = calculateBasicRelevance(paper, query);
            
            if (relevance > config.thresholds.minRelevance) {
                PaperMetadata scoredPaper = paper;
                scoredPaper.relevanceScore = relevance;
                results.push_back(scoredPaper);
            }
        }
        
        // 按相关性初步排序
        std::sort(results.begin(), results.end(),
            [](const PaperMetadata& a, const PaperMetadata& b) {
                return a.relevanceScore > b.relevanceScore;
            });
        
        return results;
    }
    
    // 计算文献综合分数
    float calculatePaperScore(
        const PaperMetadata& paper,
        const std::string& query,
        const ReaderKnowledge& reader) {
        
        float totalScore = 0.0f;
        
        // 1. 查询相关性
        totalScore += paper.relevanceScore * config.weights.relevance;
        
        // 2. 知识缺口匹配度
        float gapScore = calculateKnowledgeGapScore(paper, reader);
        paper.knowledgeGapScore = gapScore;
        totalScore += gapScore * config.weights.knowledgeGap;
        
        // 3. 难度匹配度
        float difficultyScore = calculateDifficultyMatchScore(paper, reader);
        paper.difficultyMatchScore = difficultyScore;
        totalScore += difficultyScore * config.weights.difficultyMatch;
        
        // 4. 时间效率
        float timeScore = calculateTimeEfficiencyScore(paper, reader);
        totalScore += timeScore * config.weights.timeEfficiency;
        
        // 5. 新颖性（避免推荐已掌握内容）
        float noveltyScore = calculateNoveltyScore(paper, reader);
        totalScore += noveltyScore * config.weights.novelty;
        
        // 6. 目标匹配度
        float goalScore = calculateGoalMatchScore(paper, reader);
        totalScore += goalScore * config.weights.goalMatch;
        
        // 归一化到[0, 1]
        return std::min(1.0f, std::max(0.0f, totalScore));
    }
    
    // 计算基础相关性（简化版，实际应使用TF-IDF或BERT等）
    float calculateBasicRelevance(const PaperMetadata& paper, 
                                  const std::string& query) {
        // 简单实现：检查查询词是否出现在标题或摘要中
        std::string text = paper.title + " " + paper.abstract;
        std::transform(text.begin(), text.end(), text.begin(), ::tolower);
        std::string queryLower = query;
        std::transform(queryLower.begin(), queryLower.end(), queryLower.begin(), ::tolower);
        
        size_t pos = 0;
        int count = 0;
        while ((pos = text.find(queryLower, pos)) != std::string::npos) {
            count++;
            pos += queryLower.length();
        }
        
        // 根据出现次数计算相关性
        float relevance = 1.0f - std::exp(-0.5f * count);
        return relevance;
    }
    
    // 计算知识缺口匹配度
    float calculateKnowledgeGapScore(const PaperMetadata& paper,
                                     const ReaderKnowledge& reader) {
        if (reader.knowledgeGaps.empty()) {
            return 0.5f;  // 如果没有指定知识缺口，返回中性分数
        }
        
        float totalScore = 0.0f;
        int matchCount = 0;
        
        for (const auto& gap : reader.knowledgeGaps) {
            // 检查这个知识缺口是否在文献的概念中
            auto it = paper.concepts.find(gap);
            if (it != paper.concepts.end()) {
                totalScore += it->second;  // 使用概念权重
                matchCount++;
            }
        }
        
        if (matchCount > 0) {
            return totalScore / matchCount;
        }
        
        return 0.0f;
    }
    
    // 计算难度匹配度
    float calculateDifficultyMatchScore(const PaperMetadata& paper,
                                        const ReaderKnowledge& reader) {
        int readerLevel = static_cast<int>(reader.readingLevel);
        int paperLevel = static_cast<int>(paper.difficulty);
        
        // 计算难度差
        int diff = std::abs(paperLevel - readerLevel);
        
        // 理想情况：难度略高于读者水平（+1级）
        if (diff == 1 && paperLevel > readerLevel) {
            return 1.0f;  // 最佳匹配：稍有挑战
        } else if (diff == 0) {
            return 0.8f;  // 完全匹配：适中
        } else if (diff == 1 && paperLevel < readerLevel) {
            return 0.6f;  // 稍微简单
        } else if (diff == 2) {
            return 0.3f;  // 难度差异较大
        } else {
            return 0.1f;  // 难度差异非常大
        }
    }
    
    // 计算时间效率分数
    float calculateTimeEfficiencyScore(const PaperMetadata& paper,
                                       const ReaderKnowledge& reader) {
        if (reader.timeLimit <= 0) {
            return 1.0f;  // 无时间限制
        }
        
        // 计算阅读效率：引用数/阅读时间
        float efficiency = static_cast<float>(paper.citationCount) / 
                          std::max(1.0f, paper.estimatedReadingTime);
        
        // 归一化（假设最大效率为10）
        float normalizedEfficiency = std::min(1.0f, efficiency / 10.0f);
        
        // 考虑时间限制：如果文献太长，降低分数
        float timePenalty = paper.estimatedReadingTime > reader.timeLimit ? 
                           0.5f : 1.0f;
        
        return normalizedEfficiency * timePenalty;
    }
    
    // 计算新颖性分数
    float calculateNoveltyScore(const PaperMetadata& paper,
                                const ReaderKnowledge& reader) {
        float novelty = 1.0f;
        
        for (const auto& concept : paper.concepts) {
            auto it = reader.conceptMastery.find(concept.first);
            if (it != reader.conceptMastery.end()) {
                // 如果读者已经掌握这个概念，降低新颖性
                novelty *= (1.0f - it->second * 0.5f);
            }
        }
        
        return novelty;
    }
    
    // 计算目标匹配度
    float calculateGoalMatchScore(const PaperMetadata& paper,
                                  const ReaderKnowledge& reader) {
        switch (reader.goal) {
            case ReaderKnowledge::LearningGoal::OVERVIEW:
                return paper.paperType == PaperMetadata::PaperType::REVIEW ? 1.0f : 0.3f;
                
            case ReaderKnowledge::LearningGoal::DEEP_DIVE:
                return paper.paperType == PaperMetadata::PaperType::THEORETICAL ? 1.0f : 0.4f;
                
            case ReaderKnowledge::LearningGoal::METHOD_LEARNING:
                return paper.paperType == PaperMetadata::PaperType::METHODOLOGICAL ? 1.0f : 
                       paper.paperType == PaperMetadata::PaperType::TUTORIAL ? 0.8f : 0.2f;
                
            case ReaderKnowledge::LearningGoal::PROBLEM_SOLVING:
                return paper.paperType == PaperMetadata::PaperType::EMPIRICAL ? 1.0f : 0.4f;
                
            default:
                return 0.5f;
        }
    }
    
    // 生成推荐理由
    std::vector<std::string> generateRecommendationReasons(
        const PaperMetadata& paper,
        const ReaderKnowledge& reader) {
        
        std::vector<std::string> reasons;
        
        // 基于知识缺口
        for (const auto& gap : reader.knowledgeGaps) {
            if (paper.concepts.find(gap) != paper.concepts.end()) {
                reasons.push_back("填补知识缺口: " + gap);
            }
        }
        
        // 基于难度匹配
        float diffScore = calculateDiffic
        // main.cpp
#include "retrieval_engine.h"
#include <iostream>
#include <iomanip>

using namespace KnowledgeRetrieval;

// 创建示例知识图谱
KnowledgeGraph createSampleKnowledgeGraph() {
    KnowledgeGraph graph;
    
    // 机器学习相关概念
    ConceptNode mlNode;
    mlNode.concept = "machine learning";
    mlNode.prerequisites = {"statistics", "linear algebra"};
    mlNode.relatedConcepts = {"deep learning", "supervised learning", "unsupervised learning"};
    mlNode.depth = 2;
    mlNode.importance = 0.9f;
    graph.addConcept("machine learning", mlNode);
    
    ConceptNode dlNode;
    dlNode.concept = "deep learning";
    dlNode.prerequisites = {"machine learning", "neural networks"};
    dlNode.relatedConcepts = {"CNN", "RNN", "transformer"};
    dlNode.depth = 3;
    dlNode.importance = 0.8f;
    graph.addConcept("deep learning", dlNode);
    
    return graph;
}

// 创建示例文献数据库
std::vector<PaperMetadata> createSamplePaperDatabase() {
    std::vector<PaperMetadata> papers;
    
    // 论文1：机器学习综述
    PaperMetadata paper1;
    paper1.id = "P001";
    paper1.title = "A Comprehensive Review of Machine Learning Techniques";
    paper1.abstract = "This paper provides a comprehensive review of various machine learning techniques...";
    paper1.year = 2022;
    paper1.difficulty = PaperMetadata::Difficulty::INTERMEDIATE;
    paper1.paperType = PaperMetadata::PaperType::REVIEW;
    paper1.estimatedReadingTime = 3.5f;
    paper1.citationCount = 150;
    paper1.concepts = {
        {"machine learning", 0.9f},
        {"supervised learning", 0.7f},
        {"unsupervised learning", 0.7f}
    };
    
    // 论文2：深度学习基础
    PaperMetadata paper2;
    paper2.id = "P002";
    paper2.title = "Introduction to Deep Learning";
    paper2.abstract = "This tutorial introduces the basic concepts of deep learning...";
    paper2.year = 2023;
    paper2.difficulty = PaperMetadata::Difficulty::INTRODUCTORY;
    paper2.paperType = PaperMetadata::PaperType::TUTORIAL;
    paper2.estimatedReadingTime = 2.0f;
    paper2.citationCount = 80;
    paper2.concepts = {
        {"deep learning", 0.8f},
        {"neural networks", 0.9f}
    };
    
    // 论文3：Transformer架构
    PaperMetadata paper3;
    paper3.id = "P003";
    paper3.title = "Attention Is All You Need: The Transformer Architecture";
    paper3.abstract = "We propose a new simple network architecture, the Transformer...";
    paper3.year = 2021;
    paper3.difficulty = PaperMetadata::Difficulty::ADVANCED;
    paper3.paperType = PaperMetadata::PaperType::THEORETICAL;
    paper3.estimatedReadingTime = 4.0f;
    paper3.citationCount = 3000;
    paper3.concepts = {
        {"transformer", 1.0f},
        {"attention mechanism", 0.9f},
        {"deep learning", 0.8f}
    };
    
    papers.push_back(paper1);
    papers.push_back(paper2);
    papers.push_back(paper3);
    
    // 添加更多示例论文...
    for (int i = 4; i <= 20; ++i) {
        PaperMetadata paper;
        paper.id = "P" + std::to_string(i);
        paper.title = "Sample Paper " + std::to_string(i);
        paper.abstract = "This is sample paper number " + std::to_string(i);
        paper.year = 2020 + (i % 4);
        paper.difficulty = static_cast<PaperMetadata::Difficulty>((i % 4) + 1);
        paper.paperType = static_cast<PaperMetadata::PaperType>(i % 5);
        paper.estimatedReadingTime = 1.0f + (i % 5) * 0.5f;
        paper.citationCount = i * 10;
        paper.concepts = {
            {"concept_" + std::to_string(i % 5), 0.5f + (i % 5) * 0.1f}
        };
        papers.push_back(paper);
    }
    
    return papers;
}

// 创建示例读者知识
ReaderKnowledge createSampleReaderKnowledge() {
    ReaderKnowledge reader;
    reader.userId = "user123";
    reader.familiarTopics = {"programming", "basic statistics"};
    
    // 已掌握的概念
    reader.conceptMastery = {
        {"programming", 0.8f},
        {"statistics", 0.6f},
        {"linear algebra", 0.5f}
    };
    
    // 知识缺口
    reader.knowledgeGaps = {"machine learning", "deep learning", "neural networks"};
    
    // 阅读水平和目标
    reader.readingLevel = ReaderKnowledge::ReadingLevel::INTERMEDIATE;
    reader.goal = ReaderKnowledge::LearningGoal::DEEP_DIVE;
    reader.timeLimit = 10.0f;
    reader.preferredFormats = {"pdf", "html"};
    reader.maxPapers = 10;
    
    return reader;
}

// 打印检索结果
void printRetrievalResults(const std::vector<RetrievalResult>& results) {
    std::cout << "\n========== 智能文献检索结果 ==========\n";
    std::cout << "找到 " << results.size() << " 篇相关文献\n\n";
    
    for (size_t i = 0; i < results.size(); ++i) {
        const auto& result = results[i];
        
        std::cout << "[" << i + 1 << "] " << result.paper.title << "\n";
        std::cout << "     分数: " << std::fixed << std::setprecision(3) 
                  << result.finalScore << " | ";
        
        // 显示难度
        std::string difficulty;
        switch (result.paper.difficulty) {
            case PaperMetadata::Difficulty::INTRODUCTORY: difficulty = "入门"; break;
            case PaperMetadata::Difficulty::INTERMEDIATE: difficulty = "中级"; break;
            case PaperMetadata::Difficulty::ADVANCED: difficulty = "高级"; break;
            default: difficulty = "专家"; break;
        }
        std::cout << "难度: " << difficulty << " | ";
        std::cout << "年份: " << result.paper.year << " | ";
        std::cout << "引用: " << result.paper.citationCount << " | ";
        std::cout << "阅读时间: " << result.paper.estimatedReadingTime << "小时\n";
        
        // 显示推荐理由
        if (!result.reasons.empty()) {
            std::cout << "     推荐理由:\n";
            for (const auto& reason : result.reasons) {
                std::cout << "       ? " << reason << "\n";
            }
        }
        
        // 显示前置文献
        if (!result.prerequisites.empty()) {
            std::cout << "     建议先阅读:\n";
            for (const auto& pre : result.prerequisites) {
                std::cout << "       ? " << pre << "\n";
            }
        }
        
        std::cout << std::endl;
    }
    
    // 显示统计信息
    if (!results.empty()) {
        std::cout << "\n====== 统计信息 ======\n";
        
        // 难度分布
        int intro = 0, inter = 0, adv = 0;
        for (const auto& result : results) {
            switch (result.paper.difficulty) {
                case PaperMetadata::Difficulty::INTRODUCTORY: intro++; break;
                case PaperMetadata::Difficulty::INTERMEDIATE: inter++; break;
                default: adv++; break;
            }
        }
        
        std::cout << "难度分布: 入门(" << intro << "), 中级(" << inter 
                  << "), 高级(" << adv << ")\n";
        
        // 平均分数
        float avgScore = 0;
        for (const auto& result : results) {
            avgScore += result.finalScore;
        }
        avgScore /= results.size();
        std::cout << "平均推荐分数: " << std::fixed << std::setprecision(3) 
                  << avgScore << "\n";
    }
}

int main() {
    std::cout << "=== 基于知识状态的智能文献检索系统 ===\n\n";
    
    // 1. 初始化组件
    KnowledgeGraph graph = createSampleKnowledgeGraph();
    std::vector<PaperMetadata> papers = createSamplePaperDatabase();
    ReaderKnowledge reader = createSampleReaderKnowledge();
    
    // 2. 配置检索引擎
    RetrievalConfig config;
    config.strategy = RetrievalConfig::SortStrategy::HYBRID;
    config.maxPapersPerTopic = 3;
    
    IntelligentRetrievalEngine engine(config);
    
    // 3. 设置知识图谱和文献数据库
    engine.setKnowledgeGraph(graph);
    for (const auto& paper : papers) {
        engine.addPaper(paper);
    }
    
    // 4. 执行检索
    std::string query = "machine learning deep learning techniques";
    std::cout << "查询: " << query << "\n";
    std::cout << "读者水平: 中级 | 目标: 深入学习\n\n";
    
    std::vector<RetrievalResult> results = engine.retrievePapers(query, reader);
    
    // 5. 显示结果
    printRetrievalResults(results);
    
    // 6. 交互式示例
    std::cout << "\n=== 交互式调整示例 ===\n";
    std::cout << "假设读者发现第一篇文献太难...\n";
    
    // 模拟调整：读者表示第一篇太难
    reader.knowledgeGaps.push_back("attention mechanism");  // 增加知识缺口
    reader.readingLevel = ReaderKnowledge::ReadingLevel::INTERMEDIATE;
    
    std::cout << "调整后重新检索...\n\n";
    
    // 重新检索
    results = engine.retrievePapers(query, reader);
    printRetrievalResults(results);
    
    return 0;
}

