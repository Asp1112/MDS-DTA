# 101–200位机制筛选结论

完成90%去冗、UniProt天然底物经验筛选、AlphaFold结构基序/口袋检查及机制定点对接后，共有19个候选达到综合门槛。最终先按机制证据等级选择、同等级内按原始排名，共取10个；101–200已足够，因此不扩展201–300。

## 最终10个

|排名|UniProt|蛋白|路线|证据等级|AcCoA几何|判断|
|---:|---|---|---|---|---|---|
|105|Q54IU8|Probable arylamine N-acetyltransferase 2 2.3.1.5|乒乓|A：天然反应直接相关＋完整三联体＋对接命中|3.919 Å / 89.3°|芳胺N-乙酰转移酶；UniProt标注酰基-硫酯中间体，结构中有完整Cys-His-Asp/Glu口袋，AcCoA与对氨基苯酚均命中催化区。|
|106|Q7CWE8|Homoserine O-acetyltransferase HAT 2.3.1.31|乒乓|B：小分子天然底物＋酰基半胱氨酸＋对接命中|4.200 Å / 73.8°|天然受体为小分子；UniProt标注酰基-硫酯中间体，结构中存在Cys-His-Asp/Glu样口袋，AcCoA和对氨基苯酚定点对接均达到宽松反应几何。|
|111|Q7YRG5|Arylamine N-acetyltransferase 2 2.3.1.5|乒乓|A：天然反应直接相关＋完整三联体＋对接命中|4.432 Å / 92.0°|芳胺N-乙酰转移酶；UniProt标注酰基-硫酯中间体，结构中有完整Cys-His-Asp/Glu口袋，AcCoA与对氨基苯酚均命中催化区。|
|112|A8GKF8|Homoserine O-succinyltransferase HST 2.3.1.46|乒乓|B：小分子天然底物＋酰基半胱氨酸＋对接命中|4.318 Å / 133.6°|天然受体为小分子；UniProt标注酰基-硫酯中间体，结构中存在Cys-His-Asp/Glu样口袋，AcCoA和对氨基苯酚定点对接均达到宽松反应几何。|
|126|O86309|Arylamine N-acetyltransferase NAT 2.3.1.5|乒乓|A-：天然反应直接相关＋完整三联体；AcCoA采样未命中|本轮未命中|芳胺N-乙酰转移酶天然反应与目标最接近，且UniProt标注催化半胱氨酸、结构三联体完整；对氨基苯酚可进入催化区，但本轮AcCoA构象未达到几何阈值，按对接假阴性/待复核记录。|
|131|P18440|Arylamine N-acetyltransferase 1 2.3.1.5|乒乓|A-：天然反应直接相关＋完整三联体；AcCoA采样未命中|本轮未命中|芳胺N-乙酰转移酶天然反应与目标最接近，且UniProt标注催化半胱氨酸、结构三联体完整；对氨基苯酚可进入催化区，但本轮AcCoA构象未达到几何阈值，按对接假阴性/待复核记录。|
|133|C5BF07|Homoserine O-succinyltransferase HST 2.3.1.46|乒乓|B：小分子天然底物＋酰基半胱氨酸＋对接命中|4.742 Å / 127.5°|天然受体为小分子；UniProt标注酰基-硫酯中间体，结构中存在Cys-His-Asp/Glu样口袋，AcCoA和对氨基苯酚定点对接均达到宽松反应几何。|
|145|B9JUF7|Homoserine O-acetyltransferase HAT 2.3.1.31|乒乓|B：小分子天然底物＋酰基半胱氨酸＋对接命中|4.030 Å / 87.2°|天然受体为小分子；UniProt标注酰基-硫酯中间体，结构中存在Cys-His-Asp/Glu样口袋，AcCoA和对氨基苯酚定点对接均达到宽松反应几何。|
|147|Q0BX37|Homoserine O-acetyltransferase HAT 2.3.1.31|乒乓|B：小分子天然底物＋酰基半胱氨酸＋对接命中|4.260 Å / 95.4°|天然受体为小分子；UniProt标注酰基-硫酯中间体，结构中存在Cys-His-Asp/Glu样口袋，AcCoA和对氨基苯酚定点对接均达到宽松反应几何。|
|153|P13914|Arylamine N-acetyltransferase, pineal gland isozyme NAT-3 Arylamine acetylase 2.3.1.5|乒乓|A：天然反应直接相关＋完整三联体＋对接命中|4.998 Å / 87.9°|芳胺N-乙酰转移酶；UniProt标注酰基-硫酯中间体，结构中有完整Cys-His-Asp/Glu口袋，AcCoA与对氨基苯酚均命中催化区。|

## 解释边界

- 芳胺N-乙酰转移酶若UniProt明确标注芳胺反应、酰基硫酯中间体且结构三联体完整，即使单轮AcCoA构象采样未命中，也保留为A-级候选并明确标记，不把它写成对接阳性。
- HAT/HST类只有天然底物为小分子、存在标注的酰基半胱氨酸并且定点对接达到宽松反应几何时才通过。
- HXXXD本身不是充分条件；无乒乓口袋时，必须有双底物同口袋的严格顺序反应几何。
- Vina和AlphaFold结果用于实验优先级排序，不等同于已证明的催化活性。

## 长蛋白/多肽底物排除

|排名|UniProt|蛋白|排除原因|
|---:|---|---|---|
|103|P25649|ADA histone acetyltransferase complex component 2|非催化性ADA组蛋白乙酰转移酶复合物组分；涉及蛋白质乙酰化体系|
|107|P39580|Teichoic acid D-alanyltransferase 2.3.1.-|DltC/磷壁酸大分子体系的D-丙氨酰转移，不是游离小分子乙酰受体|
|108|Q1RKI1|Octanoyltransferase 2.3.1.181|原始受体为L-lysyl-[protein]，供体为octanoyl-[ACP]|
|110|O19898|Probable octanoyltransferase 2.3.1.181|原始受体为L-lysyl-[protein]，供体为octanoyl-[ACP]|
|127|Q2GH92|Octanoyltransferase 2.3.1.181|原始受体为L-lysyl-[protein]，供体为octanoyl-[ACP]|
|146|Q9NPG8|Palmitoyltransferase ZDHHC4 2.3.1.225|蛋白半胱氨酸棕榈酰化酶；原始受体为L-cysteinyl-[protein]|
|154|O31633|Probable N-acetyltransferase YjcK 2.3.1.-|原始受体为蛋白质N端氨基酸|
|156|A4J246|Octanoyltransferase 2.3.1.181|原始受体为L-lysyl-[protein]，供体为octanoyl-[ACP]|
|157|Q04474|RTX-III toxin-activating lysine-acyltransferase ApxIIC 2.3.1.-|RTX毒素蛋白赖氨酸酰化；原始受体为L-lysyl-[protein]|
|179|Q8D326|Octanoyltransferase 2.3.1.181|原始受体为L-lysyl-[protein]，供体为octanoyl-[ACP]|
|186|Q6CJC5|Palmitoyltransferase SWF1 2.3.1.225|蛋白半胱氨酸棕榈酰化酶；原始受体为L-cysteinyl-[protein]|
|189|C0H559|Alpha-tubulin N-acetyltransferase Alpha-TAT TAT 2.3.1.108|原始受体为alpha-tubulin蛋白赖氨酸|
|198|Q6BP23|Palmitoyltransferase SWF1 2.3.1.225|蛋白半胱氨酸棕榈酰化酶；原始受体为L-cysteinyl-[protein]|
