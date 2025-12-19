from pptx import Presentation
from pptx.util import Inches, Pt
from pptx.enum.text import PP_ALIGN
from pptx.dml.color import RGBColor

def create_ppt():
    prs = Presentation()

    # --- 辅助函数：添加标题和内容 ---
    def add_slide(title_text, content_text_list):
        slide_layout = prs.slide_layouts[1] # Title and Content
        slide = prs.slides.add_slide(slide_layout)
        title = slide.shapes.title
        title.text = title_text
        
        # 设置标题字体大小
        title.text_frame.paragraphs[0].font.size = Pt(32)
        title.text_frame.paragraphs[0].font.bold = True

        body_shape = slide.shapes.placeholders[1]
        tf = body_shape.text_frame
        tf.clear()  # 清除默认空的段落

        for item in content_text_list:
            p = tf.add_paragraph()
            p.text = item
            p.font.size = Pt(20)
            # 简单的层级处理：如果以"-"开头，缩进
            if item.startswith("  -"):
                p.level = 1
                p.text = item.strip(" -")
            elif item.startswith("    *"):
                p.level = 2
                p.text = item.strip(" *")
            
            p.space_after = Pt(10)

    # --- 辅助函数：封面页 ---
    def add_cover(title, subtitle, info_dict):
        slide_layout = prs.slide_layouts[0] # Title Slide
        slide = prs.slides.add_slide(slide_layout)
        
        title_shape = slide.shapes.title
        title_shape.text = title
        
        subtitle_shape = slide.placeholders[1]
        subtitle_text = subtitle + "\n\n"
        for k, v in info_dict.items():
            subtitle_text += f"{k}：{v}\n"
        subtitle_shape.text = subtitle_text

    # --- Slide 1: 封面 ---
    add_cover(
        "主行存内存列架构下的\n行列混合查询优化方法研究",
        "研究生中期报告",
        {
            "汇报人": "谭昊",
            "学号": "M202374068",
            "导师": "张勇",
            "专业": "计算机技术",
            "日期": "202X年X月X日"
        }
    )

    # --- Slide 2: 目录 ---
    add_slide("汇报提纲", [
        "1. 课题背景与问题阐述",
        "2. 国内外研究现状",
        "3. 研究目标与核心内容",
        "4. 技术方案与系统架构",
        "5. 当前研究进展与成果",
        "6. 后续工作计划"
    ])

    # --- Slide 3: 课题背景 ---
    add_slide("1. 课题背景与问题阐述", [
        "HTAP 数据库发展趋势",
        "  - 混合事务与分析处理 (HTAP) 需求激增",
        "  - 主流架构：主行存 (OLTP) + 内存列存 (OLAP 副本)",
        "面临的挑战",
        "  - 混合行列扫描 (Hybrid Scan) 决策困难",
        "  - 挑战1：候选计划搜索空间有限，容易陷入局部最优",
        "  - 挑战2：传统代价模型在异构引擎下估算失准",
        "  - 挑战3：跨引擎拆分缺乏灵活性，常局限于根节点拆分"
    ])

    # --- Slide 4: 研究现状 ---
    add_slide("2. 国内外研究现状", [
        "智能查询优化技术 (AI4DB)",
        "  - 学习型基数估计 (LCE): DACE, ASM",
        "  - 学习型代价模型 (LCM): 替代传统手工公式",
        "  - 端到端优化器: Neo, Balsa, Lero (Learning-to-Rank)",
        "现有技术的局限性",
        "  - 大多基于单一存储引擎假设",
        "  - 缺乏对行列共存异构环境的特征表征",
        "  - 训练数据存在重复或矛盾，影响模型泛化能力"
    ])

    # --- Slide 5: 研究目标 ---
    add_slide("3. 研究目标与核心内容", [
        "核心理念",
        "  - 查询优化即服务 (QOaaS)",
        "  - 非侵入式架构：不修改数据库内核代码",
        "三大研究内容",
        "  - (1) 行列混合候选计划探索方法",
        "    * 逻辑改写 + 细粒度物理枚举",
        "  - (2) 智能计划选择与排序模型",
        "    * Tree-LSTM 网络 + 特征深度融合",
        "  - (3) 计划拆分与执行调度",
        "    * 基于 CTE 的跨引擎分发"
    ])

    # --- Slide 6: 总体架构 ---
    add_slide("4. 技术方案：总体架构", [
        "[在此处插入报告中图 4-1 AI 优化器架构图]",
        "四大核心功能模块",
        "  - 数据库交互器：特征感知与执行反馈",
        "  - 计划探索器：生成高质量候选计划集",
        "  - 计划比较器：预测并选出最优计划",
        "  - 计划拆分器：生成可执行 SQL 并分发",
        "工作流程",
        "  - SQL -> 逻辑改写 -> 物理采样 -> 特征编码 -> 排序决策 -> 拆分执行"
    ])

    # --- Slide 7: 模块细节1 ---
    add_slide("4.1 数据库交互与计划探索", [
        "数据库交互器 (DB Interactor)",
        "  - 特征提取：直方图、Sketch、列存视图、CTE信息",
        "  - 图结构构建：解析外键与索引，构建 Join Graph",
        "  - 向量化：统一异构特征供下游模型使用",
        "计划探索器 (Plan Explorer)",
        "  - 逻辑改写：解嵌套 (Unnesting) 消除数据强依赖",
        "  - 混合枚举：识别独立子树，枚举行列路径",
        "  - 带权重水塘采样：解决空间爆炸，保证样本多样性"
    ])

    # --- Slide 8: 模块细节2 ---
    add_slide("4.2 计划比较与拆分", [
        "计划比较器 (Plan Comparator)",
        "  - Tree-LSTM：对树形计划进行自底向上编码",
        "  - 特征融合：结合计划结构嵌入 + 底层统计特征",
        "  - 决策：输出预测概率最高的 Top-1 计划",
        "计划拆分器 (Plan Splitter)",
        "  - 边界识别：自动识别异构算子执行边界",
        "  - SQL 重构：利用 WITH 子句封装子查询",
        "  - 协同执行：利用内存物化机制传递中间结果"
    ])

    # --- Slide 9: 研究进展 ---
    add_slide("5. 当前研究进展 (1/2)", [
        "已完成工作",
        "  - 开源 Lero 优化器的行列场景适配",
        "  - 实现逻辑计划改写规则库（如依赖连接上拉）",
        "  - 完成 Tree-LSTM 排序模型的搭建与训练",
        "逻辑改写成果",
        "  - 成功消除关联子查询的数据依赖",
        "  - 实现了子树的独立拆分（如图 5-1 所示）",
        "  - 验证了“逻辑改写+混合枚举”策略的有效性"
    ])

    # --- Slide 10: 研究进展 ---
    add_slide("5. 当前研究进展 (2/2)", [
        "模型训练效果",
        "  - 相比原始 Lero，MSE 损失从 0.41 降至 0.28 (提升 33%)",
        "初步性能评估 (TPC-DS 基准)",
        "  - 实验环境：PostgreSQL (行) + DuckDB (列)",
        "  - 结果：在 26 个复杂查询中生成了更优的混合计划",
        "  - 优势：长耗时查询 (q14, q45) 实现数倍时延缩减",
        "  - 证明了 AI 驱动的行列混合优化方法的潜力"
    ])

    # --- Slide 11: 总结与展望 ---
    add_slide("6. 总结与后续工作", [
        "主要创新点",
        "  - 提出细粒度的行列混合计划探索策略",
        "  - 设计基于 Tree-LSTM 的跨模态特征融合方法",
        "  - 实现非侵入式的 HTAP 优化器原型",
        "后续工作计划",
        "  - 完善网络结构，增加嵌入维度",
        "  - 丰富训练数据，提升模型泛化能力",
        "  - 完成最终的系统评估与毕业论文撰写"
    ])

    # --- Slide 12: 结束 ---
    slide_layout = prs.slide_layouts[1]
    slide = prs.slides.add_slide(slide_layout)
    title = slide.shapes.title
    title.text = "致谢"
    content = slide.placeholders[1]
    content.text = "\n感谢各位老师的聆听！\n请批评指正。"
    content.text_frame.paragraphs[0].alignment = PP_ALIGN.CENTER

    # Save
    prs.save('中期报告_M202374068_谭昊.pptx')
    print("PPT生成成功！")

if __name__ == "__main__":
    create_ppt()