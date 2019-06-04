期权定价
==============
期权
-----------------
**What** is option ?  
期权，是指一种合约，该合约赋予持有人在某一特定日期或该日之前的任何时间以固定价格购进或售出一种资产的权利。  
期权定义的要点如下：  
1、期权是一种权利。 期权合约至少涉及买家和出售人两方。持有人享有权力但不承担相应的义务。  
2、期权的标的物。期权的标的物是指选择购买或出售的资产。它包括股票、政府债券、货币、股票指数、商品期货等。期权是这些标的物“衍生”的，因此称衍生金融工具。  
值得注意的是，期权出售人不一定拥有标的资产。期权是可以“卖空”的。期权购买人也不定真的想购买资产标的物。因此，期权到期时双方不一定进行标的物的实物交割，而只需按价差补足价款即可。
3、到期日。双方约定的期权到期的那一天称为“到期日”，如果该期权只能在到期日执行，则称为*欧式期权*；如果该期权可以在到期日或到期日之前的任何时间执行，则称为*美式期权*。
4、期权的执行。依据期权合约购进或售出标的资产的行为称为“执行”。在期权合约中约定的、期权持有人据以购进或售出标的资产的固定价格，称为“执行价格”。   

期权的定价
------------------

期权定价模型（OPM）----由布莱克与斯科尔斯在20世纪70年代提出。该模型认为，只有股价的当前值与未来的预测有关；变量过去的历史与演变方式与未来的预测不相关 。模型表明，期权价格的决定非常复杂，合约期限、股票现价、无风险资产的利率水平以及交割价格等都会影响期权价格。

期权定价的主要模型
------------------------
B-S模型  

期权定价模型基于对冲证券组合的思想。投资者可建立期权与其标的股票的组合来保证确定报酬。在均衡时，此确定报酬必须得到无风险利率。期权的这一定价思想与无套利定价的思想是一致的。所谓无套利定价就是说任何零投入的投资只能得到零回报，任何非零投入的投资，只能得到与该项投资的风险所对应的平均回报，而不能获得超额回报（*超过与风险相当的报酬的利润*）。从Black-Scholes期权定价模型的推导中，不难看出期权定价本质上就是无套利定价。  
**假设条件**  
*1、标的资产价格服从对数正态分布；*  
*2、在期权有效期内，无风险利率和金融资产收益变量是恒定的；*
*3、市场无摩擦，即不存在税收和交易成本；*
*4、金融资产在期权有效期内无红利及其它所得(该假设后被放弃)；*
*5、该期权是欧式期权，即在期权到期前不可实施。*

**定价公式**
C=S·N(D1)-L·(E^(-γT))*N(D2)
其中:
D1=(Ln(S/L)+(γ+(σ^2)/2)*T)/(σ*T^(1/2))
D2=D1-σ*T^(1/2)
C—期权初始合理价格
L—期权交割价格
S—所交易金融资产现价
T—期权有效期
γ—连续复利计无风险利率H
σ2—年度化方差
N()—正态分布变量的累积概率分布函数，在此应当说明两点：
第一，该模型中无风险利率必须是连续复利形式。一个简单的或不连续的无风险利率(设为γ0)一般是一年复利一次，而γ要求利率连续复利。γ0必须转化为r方能代入上式计算。两者换算关系为:γ=LN(1+γ0)或γ0=Eγ-1。例如γ0=0.06，则γ=LN(1+0.06)=0583，即100以583%的连续复利投资第二年将获106，该结果与直接用γ0=0.06计算的答案一致。
第二，期权有效期T的相对数表示，即期权有效天数与一年365天的比值。如果期权有效期为100天，则T=100/365=0.274。
**推导运用**
(一)B-S模型的推导是由看涨期权入手的，对于一项看涨期权，其到期的期值是:E[G]=E[max(ST-L，O)]
其中，E[G]—看涨期权到期期望值ST—到期所交易金融资产的市场价值
L—期权交割(实施)价
到期有两种可能情况:  
1、如果STL，则期权实施以进帐(In-the-money)生效，且mAx(ST-L，O)=ST-L
2、如果ST<>
max(ST-L，O)=0
从而:E[CT]=P×(E[ST|STL)+(1-P)×O=P×(E[ST|STL]-L)
其中:P—(STL)的概率E[ST|STL]—既定(STL)下ST的期望值将E[G]按有效期无风险连续复利rT贴现，得期权初始合理价格:C=P×E-rT×(E[ST|STL]-L)(*)这样期权定价转化为确定P和E[ST|STL]。
首先，
对收益进行定义。与利率一致，收益为金融资产期权交割日市场价格(ST)与现价(S)比值的对数值，即收益=1NSTS。由假设1收益服从对数正态分布，即1NSTS～N(μT，σT2)，所以E[1N(STS]=μT，STS～EN(μT，σT2)可以证明，相对价格期望值大于EμT，为:E[STS]=EμT+σT22=EμT+σ2T2=EγT从而，μT=T(γ-σ22)，且有σT=σT其次，求(STL)的概率P，也即求收益大于(LS)的概率。已知正态分布有性质:Pr06[ζχ]=1-N(χ-μσ)其中:ζ—正态分布随机变量χ—关键值μ—ζ的期望值σ—ζ的标准差所以:P=Pr06[ST1]=Pr06[1NSTS]1NLS]=1N-1NLS2)TTNC4由对称性:1-N(D)=N(-D)P=N1NSL+(γ-σ22)TσTArS第三，求既定STL下ST的期望值。因为E[ST|ST]L]处于正态分布的L到∞范围，所以，E[ST|ST]=S EγT N(D1)N(D2)
其中:
D1=LNSL+(γ+σ22)TσTD2=LNSL+(γ-σ22)TσT=D1-σT最后，
将P、E[ST|ST]L]代入(*)式整理得B-S定价模型:C=S N(D1)-L E-γT N(D2)(二)B-S模型应用实例假设市场上某股票现价S为　164，无风险连续复利利率γ是0.0521，市场方差σ2为0.0841，那么实施价格L是165，有效期T为0.0959的期权初始合理价格计算步骤如下:
①求D1:D1=(1N164165+(0.052)+0.08412)×0.09590.29×0.0959=0.0328
②求D2:D2=0.0328-0.29×0.0959=-0.570
③查标准正态分布函数表，得:N(0.03)=0.5120　N(-0.06)=0.4761
④求C:C=164×0.5120-165×E-0.0521×0.0959×0.4761=5.803
因此理论上该期权的合理价格是5.803。如果该期权市场实际价格是5.75，那么这意味着该期权有所低估。在没有交易成本的条件下，购买该看涨期权有利可图。
(三)看跌期权定价公式的推导B-S模型是看涨期权的定价公式。
根据*售出—购进平价理论(Put-callparity)*可以推导出有效期权的定价模型，由售出—购进平价理论，购买某股票和该股票看跌期权的组合与购买该股票同等条件下的看涨期权和以期权交割价为面值的无风险折扣发行债券具有同等价值，以公式表示为:
S+PE(S，T，L)=CE(S，T，L)+L(1+γ)-T
移项得:PE(S，T，L)=CE(S，T，L)+L(1+γ)-T-S，将B-S模型代入整理得:P=L E-γT [1-N(D2)]-S[1-N(D1)]此即为看跌期权初始价格定价模型。
**发展**
B-S模型只解决了不分红股票的期权定价问题，默顿发展了B-S模型，使其亦运用于支付红利的股票期权。(一)存在已知的不连续红利假设某股票在期权有效期内某时间T(即除息日)支付已知红利DT，只需将该红利现值从股票现价S中除去，将调整后的股票价值S′代入B-S模型中即可:S′=S-DT E-rT。如果在有效期内存在其它所得，依该法一一减去。从而将B-S模型变型得新公式:
C=(S- E-γT N(D1)-L E-γT N(D2)
(二)存在连续红利支付是指某股票以一已知分红率(设为δ)支付不间断连续红利，假如某公司股票年分红率δ为0.04，该股票现值为164，从而该年可望得红利164×0.04=6.56。值得注意的是，该红利并非分4季支付每季164；事实上，它是随美元的极小单位连续不断的再投资而自然增长的，一年累积成为6.56。因为股价在全年是不断波动的，实际红利也是变化的，但分红率是固定的。因此，该模型并不要求红利已知或固定，它只要求红利按股票价格的支付比例固定。
在此红利现值为:S(1-E-δT)，所以S′=S E-δT，以S′代S，得存在连续红利支付的期权定价公式:C=S E-δT N(D1)-L E-γT N(D2)
**影响**
自B-S模型1973年首次在政治经济杂志(Journalofpo Litical Economy)发表之后，芝加哥期权交易所的交易商们马上意识到它的重要性，很快将B-S模型程序化输入计算机应用于刚刚营业的芝加哥期权交易所。该公式的应用随着计算机、通讯技术的进步而扩展。到今天，该模型以及它的一些变形已被期权交易商、投资银行、金融管理者、保险人等广泛使用。衍生工具的扩展使国际金融市场更富有效率，但也促使全球市场更加易变。新的技术和新的金融工具的创造加强了市场与市场参与者的相互依赖，不仅限于一国之内还涉及他国甚至多国。结果是一个市场或一个国家的波动或金融危机极有可能迅速的传导到其它国家乃至整个世界经济之中。中国金融体制不健全、资本市场不完善，但是随着改革的深入和向国际化靠拢，资本市场将不断发展，汇兑制度日渐完善，企业也将拥有更多的自主权从而面临更大的风险。因此，对规避风险的金融衍生市场的培育是必需的，对衍生市场进行探索也是必要的，人们才刚刚起步。