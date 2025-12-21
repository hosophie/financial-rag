import json
import argparse
from datetime import datetime
from typing import Dict, List, Tuple
import os


class RegressionTester:
    """RAG系统回归测试工具：比较两次评估结果，检测性能退化"""
    
    # 退化阈值配置（负值表示允许下降的最大幅度）
    THRESHOLDS = {
        "faithfulness": -0.05,              # 忠实度/根据性（groundedness）
        "answer_relevancy": -0.05,
        "context_precision": -0.05,
        "context_recall": -0.05,
        "initial_retrieval_hit_rate": -0.10,  # 检索质量
        "reranked_hit_rate": -0.10,           # 检索质量
    }
    
    def __init__(self, baseline_path: str, current_path: str):
        """初始化回归测试器
        
        Args:
            baseline_path: 基准测试结果JSON文件路径
            current_path: 当前测试结果JSON文件路径
        """
        self.baseline_path = baseline_path
        self.current_path = current_path
        self.baseline_data = self._load_json(baseline_path)
        self.current_data = self._load_json(current_path)
        
    def _load_json(self, path: str) -> Dict:
        """加载JSON文件"""
        with open(path, 'r', encoding='utf-8') as f:
            return json.load(f)
    
    def _compare_metric(self, metric_name: str, baseline_val: float, 
                       current_val: float) -> Dict:
        """比较单个指标
        
        Returns:
            包含比较结果的字典
        """
        delta = current_val - baseline_val
        delta_percent = (delta / baseline_val * 100) if baseline_val != 0 else 0
        threshold = self.THRESHOLDS.get(metric_name, -0.05)
        
        # 判断是否退化：delta < threshold（即下降超过阈值）
        is_regression = delta < threshold
        
        return {
            "baseline_value": round(baseline_val, 4),
            "current_value": round(current_val, 4),
            "delta": round(delta, 4),
            "delta_percent": round(delta_percent, 2),
            "threshold": threshold,
            "status": "REGRESSION" if is_regression else "PASS",
            "severity": self._get_severity(delta, threshold)
        }
    
    def _get_severity(self, delta: float, threshold: float) -> str:
        """根据下降幅度判断严重程度"""
        if delta >= 0:
            return "IMPROVEMENT"
        elif delta >= threshold:
            return "ACCEPTABLE"
        elif delta >= threshold * 2:
            return "WARNING"
        else:
            return "CRITICAL"
    
    def _compare_summary_metrics(self) -> Dict:
        """比较汇总指标"""
        baseline_metrics = self.baseline_data["summary_metrics"]
        current_metrics = self.current_data["summary_metrics"]
        
        comparison = {}
        for metric_name in self.THRESHOLDS.keys():
            if metric_name in baseline_metrics and metric_name in current_metrics:
                comparison[metric_name] = self._compare_metric(
                    metric_name,
                    baseline_metrics[metric_name],
                    current_metrics[metric_name]
                )
        
        # 添加延迟比较（仅供参考，不影响回归判定）
        if "average_end_to_end_latency_seconds" in baseline_metrics:
            baseline_latency = baseline_metrics["average_end_to_end_latency_seconds"]
            current_latency = current_metrics["average_end_to_end_latency_seconds"]
            comparison["latency"] = {
                "baseline_value": round(baseline_latency, 4),
                "current_value": round(current_latency, 4),
                "delta": round(current_latency - baseline_latency, 4),
                "status": "INFO"
            }
        
        return comparison
    
    def _compare_per_question(self) -> List[Dict]:
        """逐问题比较详细结果"""
        baseline_details = self.baseline_data.get("detailed_results", [])
        current_details = self.current_data.get("detailed_results", [])
        
        per_question_comparison = []
        
        # 按问题匹配（假设顺序一致）
        for i, (baseline_item, current_item) in enumerate(zip(baseline_details, current_details)):
            question = baseline_item["question"]
            
            # 比较关键指标
            baseline_metrics = baseline_item.get("metrics", {})
            current_metrics = current_item.get("metrics", {})
            
            metrics_diff = {}
            for metric_name in ["faithfulness", "answer_relevancy", "context_precision", "context_recall"]:
                if metric_name in baseline_metrics and metric_name in current_metrics:
                    b_val = baseline_metrics[metric_name]
                    c_val = current_metrics[metric_name]
                    metrics_diff[metric_name] = {
                        "baseline": round(b_val, 4),
                        "current": round(c_val, 4),
                        "delta": round(c_val - b_val, 4)
                    }
            
            # 检索命中率对比
            baseline_hit = baseline_item.get("hit_analysis", {})
            current_hit = current_item.get("hit_analysis", {})
            
            hit_diff = {
                "initial_retrieval_hit": {
                    "baseline": baseline_hit.get("initial_retrieval_hit", False),
                    "current": current_hit.get("initial_retrieval_hit", False),
                    "changed": baseline_hit.get("initial_retrieval_hit") != current_hit.get("initial_retrieval_hit")
                },
                "reranked_hit": {
                    "baseline": baseline_hit.get("reranked_hit", False),
                    "current": current_hit.get("reranked_hit", False),
                    "changed": baseline_hit.get("reranked_hit") != current_hit.get("reranked_hit")
                }
            }
            
            per_question_comparison.append({
                "question_index": i,
                "question": question,
                "metrics_comparison": metrics_diff,
                "hit_analysis_comparison": hit_diff
            })
        
        return per_question_comparison
    
    def _generate_overall_status(self, summary_comparison: Dict) -> str:
        """根据汇总指标比较结果生成整体状态"""
        regression_count = sum(1 for v in summary_comparison.values() 
                              if isinstance(v, dict) and v.get("status") == "REGRESSION")
        
        if regression_count == 0:
            return "PASS"
        elif regression_count <= 2:
            return "WARNING"
        else:
            return "FAIL"
    
    def run_regression_test(self) -> Dict:
        """执行回归测试并返回完整结果"""
        print("正在执行回归测试...")
        print(f"  基准文件: {self.baseline_path}")
        print(f"  当前文件: {self.current_path}")
        
        # 比较汇总指标
        summary_comparison = self._compare_summary_metrics()
        
        # 比较详细结果
        per_question_comparison = self._compare_per_question()
        
        # 生成整体状态
        overall_status = self._generate_overall_status(summary_comparison)
        
        # 统计信息
        baseline_samples = self.baseline_data["summary_metrics"].get("total_samples", 0)
        current_samples = self.current_data["summary_metrics"].get("total_samples", 0)
        
        # 构建回归测试报告
        report = {
            "test_metadata": {
                "baseline_file": self.baseline_path,
                "current_file": self.current_path,
                "test_timestamp": datetime.now().isoformat(),
                "baseline_samples": baseline_samples,
                "current_samples": current_samples,
                "overall_status": overall_status
            },
            "configuration_comparison": {
                "baseline_config": self.baseline_data.get("experiment_config", {}),
                "current_config": self.current_data.get("experiment_config", {})
            },
            "summary_metrics_comparison": summary_comparison,
            "regression_summary": self._generate_regression_summary(summary_comparison),
            "per_question_comparison": per_question_comparison,
            "thresholds_used": self.THRESHOLDS
        }
        
        return report
    
    def _generate_regression_summary(self, summary_comparison: Dict) -> Dict:
        """生成回归汇总信息"""
        regressions = []
        warnings = []
        improvements = []
        
        for metric_name, comparison in summary_comparison.items():
            if not isinstance(comparison, dict) or "status" not in comparison:
                continue
                
            status = comparison["status"]
            severity = comparison.get("severity", "")
            
            if status == "REGRESSION":
                regressions.append({
                    "metric": metric_name,
                    "severity": severity,
                    "delta": comparison.get("delta", 0),
                    "delta_percent": comparison.get("delta_percent", 0)
                })
            elif severity == "WARNING":
                warnings.append({
                    "metric": metric_name,
                    "delta": comparison.get("delta", 0)
                })
            elif severity == "IMPROVEMENT":
                improvements.append({
                    "metric": metric_name,
                    "delta": comparison.get("delta", 0)
                })
        
        return {
            "total_regressions": len(regressions),
            "total_warnings": len(warnings),
            "total_improvements": len(improvements),
            "regressions": regressions,
            "warnings": warnings,
            "improvements": improvements
        }
    
    def save_report(self, output_path: str):
        """执行测试并保存报告"""
        report = self.run_regression_test()
        
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(report, f, ensure_ascii=False, indent=2)
        
        print(f"\n回归测试完成！")
        print(f"{'='*60}")
        print(f"整体状态: {report['test_metadata']['overall_status']}")
        print(f"{'='*60}")
        
        regression_summary = report['regression_summary']
        print(f"检测到的问题:")
        print(f"  - 回归项: {regression_summary['total_regressions']}")
        print(f"  - 警告项: {regression_summary['total_warnings']}")
        print(f"  - 改进项: {regression_summary['total_improvements']}")
        
        if regression_summary['regressions']:
            print(f"\n回归详情:")
            for reg in regression_summary['regressions']:
                print(f"  ✗ {reg['metric']}: {reg['delta']:+.4f} ({reg['delta_percent']:+.2f}%) [{reg['severity']}]")
        
        if regression_summary['improvements']:
            print(f"\n改进详情:")
            for imp in regression_summary['improvements']:
                print(f"  ✓ {imp['metric']}: {imp['delta']:+.4f}")
        
        print(f"\n报告已保存至: {output_path}")
        print(f"{'='*60}\n")


def main():
    parser = argparse.ArgumentParser(
        description="RAG系统回归测试工具：比较两次评估结果并检测性能退化"
    )
    parser.add_argument(
        "--baseline", 
        type=str, 
        required=True, 
        help="基准测试结果JSON文件路径（旧版本）"
    )
    parser.add_argument(
        "--current", 
        type=str, 
        required=True, 
        help="当前测试结果JSON文件路径（新版本）"
    )
    parser.add_argument(
        "--output", 
        type=str, 
        default=None,
        help="输出报告路径（默认: output/regression_test_TIMESTAMP.json）"
    )
    
    args = parser.parse_args()
    
    # 生成默认输出路径
    if args.output is None:
        os.makedirs("output", exist_ok=True)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        args.output = f"output/regression_test_{timestamp}.json"
    
    # 执行回归测试
    tester = RegressionTester(args.baseline, args.current)
    tester.save_report(args.output)


if __name__ == "__main__":
    main()

