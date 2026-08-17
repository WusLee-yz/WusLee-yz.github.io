---
title: "Oligopolies & Game Theory"
published: 2026-07-06
description: ""
category: "学校上的课"
draft: false
tags:
  - microeconomics
  - oligopoly
  - game-theory
  - imperfect-competition
date: 2026-07-06
course: Microeconomics
---

# Oligopolies & Game Theory

本章研究寡头 (Oligopoly) 以及理解寡头行为所需的博弈论 (Game Theory)。与完全竞争 (Perfect Competition) 不同，寡头市场中企业数量很少，每个企业的决策都会显著影响市场价格和其他企业利润；与垄断 (Monopoly) 不同，寡头企业又必须持续考虑竞争者会如何反应。因此，本章的核心不是单个企业如何在真空中定价，而是企业之间的相互依赖 (Interdependence)。

![[Assets/slide_1.png]]
![[Assets/slide_2.png]]

## 1. Oligopoly 的含义与市场位置

> [!info] 核心定义：寡头 (Oligopoly)
> Oligopoly 是一种只有少数生产者的市场结构。行业中没有单个企业完全垄断市场，但每个主要企业都能影响市场价格，因此属于不完全竞争 (Imperfect Competition)。寡头市场中的企业称为寡头企业 (Oligopolist)。

Oligopoly 与垄断竞争 (Monopolistic Competition) 都属于 Imperfect Competition：企业有竞争者，但仍拥有市场力量 (Market Power)。它通常由与垄断相似的力量形成，例如规模经济、进入壁垒、固定成本、关键资源控制或网络效应；只是这些力量不足以让一家企业独占市场，而是让少数企业占据主要份额。

![[Assets/slide_3.png]]

衡量市场集中度的常用指标是 Herfindahl-Hirschman Index (HHI)。它把行业中每家企业的市场份额平方后相加。HHI 越大，说明市场越集中，竞争越弱。

> [!info] 核心定义：Herfindahl-Hirschman Index (HHI)
> HHI = sum of the squares of each firm's share of market sales。若三家企业市场份额分别为 60%, 25%, 15%，则 HHI = 60^2 + 25^2 + 15^2 = 4,450。一般而言，HHI < 1000 表示竞争性较强，HHI > 1800 常被视为寡头特征明显。

| Market Shares | HHI Calculation | Interpretation |
| --- | --- | --- |
| 60%, 25%, 15% | 60^2 + 25^2 + 15^2 = 4,450 | 市场高度集中，竞争较弱 |
| HHI < 1000 | - | Competitive industry |
| HHI > 1800 | - | Oligopoly |

![[Assets/slide_4.png]]

## 2. Strategic Interaction：寡头为何难以分析

在完全竞争或垄断模型中，企业主要根据自身成本与需求曲线决策；但在 Oligopoly 中，企业利润强烈依赖其他大企业采取什么行动。这种情形称为战略互动 (Strategic Interaction)：任何一家企业的产量、价格、广告或投资决策，都会显著改变其他企业的利润。

> [!info] 核心定义：战略互动 (Strategic Interaction)
> Strategic Interaction 指一个经济主体的最优行动取决于其他经济主体会如何行动。在寡头市场中，企业不能只问“我想怎么做”，还必须问“竞争者会如何回应”。

最简单的寡头是双头垄断 (Duopoly)，即行业中只有两家企业。Duopoly 能把 Oligopoly 的关键逻辑压缩到最清楚的形式：如果两家企业都限制产量，市场价格可以保持较高；但每家企业又都有动机偷偷多生产，以抢占更大市场份额。

![[Assets/slide_5.png]]

## 3. Collusion, Cartel 与 Noncooperative Behavior

两家或少数几家企业如果意识到“大家都少生产会更赚钱”，就可能尝试合谋 (Collusion)。合谋的最强形式是卡特尔 (Cartel)：多个生产者正式同意限制产量或维持高价格，从而提高共同利润。若寡头企业能像一个垄断者那样行动，设置垄断价格 (Monopoly Price, P_M) 并分配垄断产量 (Monopoly Quantity, Q_M)，它们就能从市场中获得最高联合利润。

> [!info] 核心定义：合谋 (Collusion) 与卡特尔 (Cartel)
> Collusion 指企业合作以避免相互压低利润，通常表现为维持高价或限制产量。Cartel 是 Collusion 的强形式，即多个生产者达成协议，遵守产量或价格限制，以提高联合利润。

![[Assets/slide_6.png]]

但合谋本身很不稳定。即使企业加入 Cartel，也有动机作弊 (Cheat)：在其他企业限制产量时，自己多生产一点，就能以较高价格卖出更多数量，短期利润更高。这种只追求自身利润、忽视自己行为对其他企业利润影响的行为，称为非合作行为 (Noncooperative Behavior)。

> [!warning] 考点：寡头的根本张力
> Oligopolists 共同看，最想达到 Monopoly Outcome；单独看，每家企业都想扩大产量或降价抢份额。因此寡头市场常在 Successful Collusion 与 Noncooperative Behavior 之间摇摆。

![[Assets/slide_7.png]]

## 4. Quantity Competition 与 Price Competition

寡头企业的竞争方式取决于短期是否存在产能约束 (Capacity Constraints)。如果企业短期内不能随意扩大产量，竞争更像数量竞争 (Quantity Competition)，也就是 Cournot Model 的基本逻辑。产能受限时，企业较容易避免过度竞争，因为每家企业都无法无限抢占市场；它们可以在没有正式协议的情况下形成某种“分市场”的结果，价格高于边际成本 (Marginal Cost, MC)，并获得利润。

> [!info] 核心概念：Quantity Competition / Cournot Model
> Quantity Competition 指企业主要选择产量而不是价格进行竞争。Cournot Model 的核心直觉是：当产能受限时，企业较容易维持价格高于 MC 的结果，但利润通常低于正式 Cartel 下的垄断利润。

![[Assets/slide_8.png]]

如果企业生产的是完全替代品 (Perfect Substitutes)，且有足够产能满足市场需求，则竞争更像价格竞争 (Price Competition)，也就是 Bertrand Model 的逻辑。每家企业都有动机略微低于竞争者价格来抢走需求，于是相互降价会持续到价格接近 MC，结果类似完全竞争。

> [!info] 核心概念：Price Competition / Bertrand Model
> Price Competition 指企业主要通过价格竞争。Bertrand Model 的关键结论是：若产品完全同质且企业产能不受限，Undercutting 会把价格推向 Marginal Cost。

![[Assets/slide_9.png]]

| Competition Mode | Key Assumption | Typical Result |
| --- | --- | --- |
| Quantity Competition (Cournot Model) | 短期产能受限 | 价格可高于 MC，利润为正，但低于 Cartel 利润 |
| Price Competition (Bertrand Model) | 产品同质且产能充足 | 企业相互压价，Price approaches MC |

## 5. 利润最大化：共同利益与个体诱因的冲突

如果寡头企业能够合作，它们会选择 Monopoly Outcome：设置 $P_M$，并把 $Q_M$ 分配给各成员。问题是，每家企业从自身角度看，都有诱因提高产量，获取更大市场份额。这样一来，总产量上升，市场价格下降，所有企业的利润都降低。

随着寡头行业中的卖方数量增加，市场结果会越来越接近竞争性市场：价格接近 MC，产量接近社会有效率水平，利润接近零。因此，寡头企业越少、产品越简单、产能越稳定，合谋越容易；企业越多，维持高价越困难。

![[Assets/slide_10.png]]

## 6. Game Theory 与 Payoff Matrix

> [!info] 核心定义：博弈论 (Game Theory)
> Game Theory 是研究相互依赖情境下行为的理论工具。它用于预测寡头这类战略环境中的结果，因为每个参与者的收益不仅取决于自己的行动，也取决于其他参与者的行动。

在博弈中，参与者获得的收益称为收益 (Payoff)。双人博弈常用收益矩阵 (Payoff Matrix) 表示：矩阵中每个格子对应双方行动组合下各自的收益。Payoff Matrix 的价值在于，它把 Interdependence 变成可以逐格比较的结构。

![[Assets/slide_11.png]]
![[Assets/slide_12.png]]

### ADM 与 Ajinomoto 的 Duopoly Payoff Matrix

课件中的 ADM 与 Ajinomoto 例子研究两家赖氨酸生产商的产量选择。两家都选择低产量时，各自利润为 180 million；若一方高产、一方低产，高产者可赚 200 million，低产者只赚 150 million；若双方都高产，各自只有 160 million。

| ADM \ Ajinomoto | Produce 30 million pounds | Produce 40 million pounds |
| --- | --- | --- |
| Produce 30 million pounds | ADM: $180 million; Ajinomoto: $180 million | ADM: $150 million; Ajinomoto: $200 million |
| Produce 40 million pounds | ADM: $200 million; Ajinomoto: $150 million | ADM: $160 million; Ajinomoto: $160 million |

这个矩阵揭示了寡头困境：从共同利益看，两家都少生产最好；但从个体利益看，无论对方选择什么，自己多生产都更有吸引力。

![[Assets/slide_13.png]]

## 7. Dominant Strategy, Prisoners' Dilemma 与 Nash Equilibrium

> [!info] 核心定义：占优策略 (Dominant Strategy)
> Dominant Strategy 是无论对方采取什么行动，自己都最优的策略。不是每个博弈都有 Dominant Strategy；是否存在取决于 Payoff Matrix 的具体收益结构。

当每个参与者的 Dominant Strategy 都是“背叛”或“作弊”，但双方都这样做反而比合作更差时，这个博弈就是囚徒困境 (Prisoners' Dilemma)。它有两个特征：第一，每个参与者都有动机选择让自己受益、让对方受损的行动；第二，当所有人都这样做时，所有人都比合作时更糟。

> [!info] 核心定义：囚徒困境 (Prisoners' Dilemma)
> Prisoners' Dilemma 是一种博弈：个体理性行动导致集体较差结果。在寡头市场中，它解释了为什么企业明知合作能提高共同利润，却仍可能选择扩大产量或降价竞争。

![[Assets/slide_14.png]]

经典 Prisoners' Dilemma 中，两名囚犯分别决定是否认罪 (Confess)。两人都不认罪时刑期较轻；若一人认罪、一人不认罪，认罪者获得轻判，不认罪者重判；若两人都认罪，双方都得到中等偏重的刑期。共同利益要求两人都不认罪，但个体诱因推动每个人认罪。

| Louise \ Thelma | Don't confess | Confess |
| --- | --- | --- |
| Don't confess | Louise: 2 years; Thelma: 2 years | Louise: 20 years; Thelma: 5 years |
| Confess | Louise: 5 years; Thelma: 20 years | Louise: 15 years; Thelma: 15 years |

![[Assets/slide_15.png]]

> [!info] 核心定义：纳什均衡 (Nash Equilibrium)
> Nash Equilibrium，也称非合作均衡 (Noncooperative Equilibrium)，是指在给定其他参与者行动的情况下，每个参与者都选择使自己收益最大化的行动，且没有人愿意单方面改变策略。

在寡头市场中，如果企业只追求自身利润、忽视相互依赖，就会落入 Noncooperative Equilibrium。这个结果未必是共同利润最大的结果，但它是每个企业在给定对方选择时不愿单独改变的结果。

![[Assets/slide_16.png]]

## 8. Multiple Nash Equilibrium 与课堂练习

并非所有博弈都只有一个 Nash Equilibrium，也并非所有参与者都有 Dominant Strategy。课件中的广告博弈显示，Firm A 和 Firm B 是否做广告，要看对方是否做广告；某些情况下，最佳反应会形成多个稳定结果。

| Firm A \ Firm B | Advertise | Don't Advertise |
| --- | --- | --- |
| Advertise | A: $1800; B: $2200 | A: $2000; B: $2400 |
| Don't Advertise | A: $1500; B: $2000 | A: $2200; B: $2600 |

> [!question] Class Exercise：广告博弈
> 如果 Firm A advertises，Firm B 应该怎么做？如果 Firm A does not advertise，Firm B 又应该怎么做？Firm B 是否拥有 Dominant Strategy？这类问题的关键是逐列或逐行比较 Payoff，而不是凭直觉判断广告是否“有用”。

![[Assets/slide_17.png]]

军备竞赛 (Arms Race) 也可以用 Prisoners' Dilemma 分析。双方共同看，可能都不建造导弹更好；但如果一方担心对方建造而自己不建造，就会选择建造。结果可能是双方都建造导弹，资源被浪费，但任何一方单独停止都不划算。

> [!question] Class Exercise：Is the Arms Race a Prisoners' Dilemma?
> 分析步骤：先找出 Nikita 和 Margaret 的总收益最大行动组合，再找出各自个体收益最大行动，最后判断 Nash (Noncooperative) Equilibrium。如果不合作，典型预测是双方都会 build the missile。

![[Assets/slide_18.png]]

## 9. Repeated Interaction, Tit for Tat 与 Tacit Collusion

一次性 Prisoners' Dilemma 往往走向非合作均衡；但如果博弈重复进行，参与者会考虑未来。企业可能牺牲短期利润来影响对方未来行为，这称为战略行为 (Strategic Behavior)。

> [!info] 核心定义：针锋相对 (Tit for Tat)
> Tit for Tat 是重复博弈中的策略：第一期先合作，之后模仿对方上一期行动。对方合作时继续合作；对方作弊时下一期惩罚。它既奖励合作，也惩罚背叛。

> [!info] 核心定义：默契合谋 (Tacit Collusion)
> Tacit Collusion 指企业没有正式协议或面对面协商，却通过限制产量、维持高价等方式提高彼此利润。它常由重复互动和相互惩罚机制支持。

![[Assets/slide_19.png]]

在 ADM 与 Ajinomoto 的重复博弈中，如果双方都采用 Tit for Tat，每年各赚 180 million。若一方永远作弊，第一年可能赚 200 million，但之后会被对方惩罚，长期利润降到 160 million。只要企业足够重视未来收益，Tit for Tat 就能让 Tacit Collusion 更稳定。

| ADM \ Ajinomoto | Tit for Tat | Always Cheat |
| --- | --- | --- |
| Tit for Tat | ADM: $180 million each year; Ajinomoto: $180 million each year | ADM: $150 million first year, then $160 million; Ajinomoto: $200 million first year, then $160 million |
| Always Cheat | ADM: $200 million first year, then $160 million; Ajinomoto: $150 million first year, then $160 million | ADM: $160 million each year; Ajinomoto: $160 million each year |

![[Assets/slide_20.png]]

## 10. Antitrust Policy 与现实中的寡头

现实中的 Oligopoly 受到法律约束。许多国家禁止企业就价格、产量或不竞争达成正式协议。在美国，这类法律通常称为反托拉斯法 (Antitrust Law)，政府防止寡头企业像垄断者一样行动的政策称为反托拉斯政策 (Antitrust Policy)。

> [!info] 核心定义：反托拉斯政策 (Antitrust Policy)
> Antitrust Policy 是政府为阻止寡头企业通过合谋、限制竞争或垄断化行为损害市场竞争而采取的政策。它通常禁止正式 Cartel 和明确的价格或产量协议。

![[Assets/slide_21.png]]

不过，即使正式合谋违法，Tacit Collusion 仍可能存在，因为企业不一定需要公开协议就能形成“不要互相打价格战”的默契。Tacit Collusion 的限制因素包括：企业数量多、产品复杂、价格方案复杂、买方议价能力强，以及企业之间利益冲突明显。

| Factor Limiting Tacit Collusion     | 中文解释                    |
| ----------------------------------- | ----------------------- |
| Large numbers of firms              | 企业越多，监控和惩罚作弊越难          |
| Complex products and pricing scheme | 产品和价格越复杂，越难判断谁在降价或作弊    |
| Bargaining power of buyers          | 大买方可迫使企业给出折扣，破坏默契       |
| Conflicts of interest among firms   | 企业成本、产能和市场目标不同，难以维持共同策略 |

## 11. Product Differentiation, Price Leadership 与 Price War

为了避免直接价格竞争，寡头企业常使用产品差异化 (Product Differentiation) 和非价格竞争 (Nonprice Competition)。例如牙膏产品实际差异可能有限，但企业会通过品牌、包装、口味、功能声称和广告说服消费者认为产品不同。

![[Assets/slide_22.png]]

> [!info] 核心定义：价格领导 (Price Leadership)
> Price Leadership 指一家企业先设定价格，其他企业随后跟随。它是一种可能支持 Tacit Collusion 的协调方式，因为企业不用公开协议，也能减少价格竞争。

> [!info] 核心定义：非价格竞争 (Nonprice Competition)
> Nonprice Competition 指企业避免直接降价，而通过广告、品牌、产品差异化、服务、包装或渠道等方式争夺消费者。

当 Tacit Collusion 破裂，或默契协议无法维持时，市场可能爆发价格战 (Price War)。在 Bertrand Competition 下，如果产品没有差异或属于大宗商品 (Commodities)，Price War 可能把利润压到接近零。

> [!warning] 考点：Price War 的触发条件
> Price War 更可能发生在产品同质、企业产能充足、Tacit Collusion 难以监控或某家企业有强烈扩张动机时。Product Differentiation 和 Price Leadership 都是寡头企业避免 Price War 的常见方式。

![[Assets/slide_23.png]]
![[Assets/slide_24.png]]

## 12. Kinked Demand Curve 与价格黏性

课件总结中提到折弯需求曲线 (Kinked Demand Curve)。它说明寡头企业即使面对一定范围内的边际成本变化，也可能不调整价格和产量，因为涨价可能导致竞争者不跟随、销量大幅下降；降价又可能引发竞争者跟随，导致价格战并破坏 Tacit Collusion。这个模型强调了寡头市场中价格黏性 (Price Rigidity) 的来源。

> [!info] 核心概念：折弯需求曲线 (Kinked Demand Curve)
> Kinked Demand Curve 用来解释寡头价格为什么可能保持稳定：企业担心单独涨价会失去顾客，也担心降价会引发竞争者报复，因此在某些成本变化范围内选择维持原价。

## 13. 本章总结

Oligopoly 是只有少数卖方的市场结构，常见且重要。它与 Monopoly 的形成原因相似，但表现为少数企业而非一家企业拥有 Market Power。寡头行为难以预测，因为企业既可以通过 Cartel 或 Tacit Collusion 提高共同利润，也可能因个体诱因而进入 Noncooperative Behavior。

![[Assets/slide_25.png]]

Game Theory 提供了分析 Oligopoly 的工具。Payoff Matrix 展示了每个参与者的收益如何依赖双方行动；Dominant Strategy、Prisoners' Dilemma 和 Nash Equilibrium 则解释了为什么个体理性可能导致集体较差结果。

![[Assets/slide_26.png]]
![[Assets/slide_27.png]]

重复互动改变了博弈结构。若企业预期未来还会相互竞争，Tit for Tat 这类策略可以通过奖励合作和惩罚背叛支持 Tacit Collusion。但现实中 Antitrust Policy 会禁止正式合谋，同时企业数量、产品复杂性、买方议价能力和利益冲突也会削弱默契合谋。

![[Assets/slide_28.png]]
![[Assets/slide_29.png]]

## 14. Key Terms

| Term | 中文对应 | 记忆要点 |
| --- | --- | --- |
| Oligopoly | 寡头 | 少数企业控制市场，企业间相互依赖 |
| Oligopolist | 寡头企业 | 寡头市场中的生产者 |
| Imperfect Competition | 不完全竞争 | 有竞争者但仍有 Market Power |
| Duopoly | 双头垄断 | 只有两家企业的 Oligopoly |
| Duopolist | 双头垄断企业 | Duopoly 中的企业 |
| Collusion | 合谋 | 合作维持高价或限制产量 |
| Cartel | 卡特尔 | 正式限制价格或产量的合谋组织 |
| Noncooperative Behavior | 非合作行为 | 忽视自身行为对他人利润的影响 |
| Interdependence | 相互依赖 | 一方收益取决于多方行动 |
| Game Theory | 博弈论 | 分析战略互动的工具 |
| Payoff | 收益 | 博弈参与者获得的利润或效用 |
| Payoff Matrix | 收益矩阵 | 展示双方行动组合下收益 |
| Prisoners' Dilemma | 囚徒困境 | 个体理性导致集体较差结果 |
| Dominant Strategy | 占优策略 | 无论对方怎么做都最优的策略 |
| Nash Equilibrium | 纳什均衡 | 给定他人行动时无人愿意单方面改变 |
| Noncooperative Equilibrium | 非合作均衡 | 不考虑外部影响时形成的 Nash Equilibrium |
| Strategic Behavior | 战略行为 | 为影响未来反应而采取的行动 |
| Tit for Tat | 针锋相对 | 先合作，再模仿对方上一期行动 |
| Tacit Collusion | 默契合谋 | 无正式协议但共同维持高利润 |
| Antitrust Policy | 反托拉斯政策 | 防止寡头像垄断者一样行动 |
| Price War | 价格战 | 合谋破裂后相互降价竞争 |
| Product Differentiation | 产品差异化 | 让消费者认为产品不同 |
| Price Leadership | 价格领导 | 一家先定价，其他企业跟随 |
| Nonprice Competition | 非价格竞争 | 用广告、品牌和服务竞争而非直接降价 |

![[Assets/slide_30.png]]
