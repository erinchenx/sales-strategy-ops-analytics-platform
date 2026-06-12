# -*- coding: utf-8 -*-
"""Sales Strategy & Operations Analytics Platform"""
import streamlit as st
import pandas as pd
import numpy as np
import requests
import plotly.graph_objects as go
import math
from pathlib import Path
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler

st.set_page_config(page_title="市场战略及资源配置分析平台", layout="wide",
                   initial_sidebar_state="expanded")

# ── 颜色
C_TITLE  = "#607D8B"
C_BLUE   = "#1565C0"
C_PURPLE = "#6A1B9A"
C_BROWN  = "#6D4C41"
C_GMKT   = "#2E7D32"
C_TEAL   = "#4A7C7A"
C_COCOA  = "#A0846A"
C_PARCH  = "#E8D9C8"
# 势能效能象限（省域财力与业绩分析）
C_QG     = "#2E7D32"   # 成熟粮仓 右上
C_QB     = "#1565C0"   # 深耕奇迹 左上
C_QBROWN = "#6D4C41"   # 边缘维持 左下
C_QRED   = "#C62828"   # 高潜低收 右下
# 人员象限（市场人员作战画像分析）
C_PB     = "#1565C0"   # 开疆拓土  左上
C_PG     = "#2E7D32"   # 战略攻坚  右上
C_PBROWN = "#6D4C41"   # 潜伏培育  左下
C_PRED   = "#C62828"   # 效能预警  右下


FONT = "PingFang SC,Hiragino Sans GB,Microsoft YaHei,sans-serif"
TIER_COLORS = {"前1/3":"#2E7D32","中1/3":"#E8A000","后1/3":"#C62828","无业绩":"#BDBDBD"}
MKT_COLORS  = {"大客户维护型":C_BLUE,"优质核心市场":C_PURPLE,"边缘待评估市场":C_BROWN,"散户扩张型":C_GMKT}
POTENCY_COLORS = {"深耕奇迹区":C_QB,"成熟粮仓区":C_QG,"边缘维持区":C_QBROWN,"高潜低收区":C_QRED}
PERSON_COLORS  = {"尖兵型":C_PB,"重炮型":C_PG,"潜伏型":C_PBROWN,"失血型":C_PRED,"待观察":"#9E9E9E"}

# 四类成交模式说明
TYPE_4_DESCS = [
    ("大客户维护型", C_BLUE,   "高价低频 (High Value, Low Frequency)",
     "典型「二八定律」贡献者。单次成交金额大，决策周期较长。",
     "策略：深度运营，建立长期信任，防止流失。"),
    ("优质核心市场", C_PURPLE, "高价高频 (High Value, High Frequency)",
     "企业的「现金奶牛」和护城河，既有忠诚度又有极高贡献值。",
     "策略：资源倾斜，投入顶级资源防御。"),
    ("边缘待评估市场",C_BROWN, "低价低频 (Low Value, Low Frequency)",
     "投入产出比（ROI）较低，可能获客不精准或产品匹配度差。",
     "策略：重新评估，考虑策略性放弃或低成本维持。"),
    ("散户扩张型",  C_GMKT,  "低价高频 (Low Value, High Frequency)",
     "极强市场渗透力，虽单笔利润薄但规模效应明显。",
     "策略：规模覆盖，自动化工具+标准化流程，追求边际效益最大化。"),
]

TYPE_DESC_MAP = {
    name: {"sub": sub, "desc": desc, "strat": strat, "color": color}
    for name, color, sub, desc, strat in TYPE_4_DESCS
}

# ── 人员画像分类函数（9种画像，基于战场偏好重心坐标）
def get_persona(x, y, radius=0.25):
    """根据加权重心坐标 (x,y) 判定人员画像类型。x: 财力偏向, y: 转化效率"""
    dist = math.sqrt(x**2 + y**2)
    if dist <= radius:
        return {"label":"平衡稳健型","icon":"⚪",
                "definition":'战场分布均匀，无明显偏好区域，展业模式具备一定的跨环境适应性，但需结合业绩分布情况判断是"全面开花"还是"全面平庸"。',
                "management":['结合总业绩水平二次判断：高业绩者是跨区域调度的理想人选；低业绩者需警惕是否陷入"哪里都做、哪里都不深"的分散陷阱',"可作为新市场探路的候选，观察其在陌生环境下的适应速度"],
                "color":"#A569BD"}
    if abs(x) <= radius and y > radius:
        return {"label":"全域高效型","icon":"⬆️",
                "definition":"无论选择高财力还是低财力省份展业，绝对产出均处于全司高位，展业判断力与执行力兼备。",
                "management":["话术与产品组合具备普适性，优先作为内训课件和标准化作业的素材来源","公司核心利润贡献者，激励机制应保持连续性，避免因晋升通道不畅导致流失","重点观察其能否带教复制，判断是否具备管理潜力"],
                "color":"#8E44AD"}
    elif abs(x) <= radius and y < -radius:
        return {"label":"全域待提升型","icon":"⬇️",
                "definition":"无论所在区域财政条件优劣，个人绝对产出持续处于全司低位，需排查是展业省份选择问题还是执行环节问题。",
                "management":['深度排查获客、跟进、促成各环节，定位具体断点，而非笼统归因于"能力不足"',"区分新人与老员工：新人重点看学习曲线斜率，老员工若长期无改善则考虑岗位适配性评估",'避免持续安排高价值资源区域，防止资源浪费'],
                "color":"#F39C12"}
    elif abs(y) <= radius and x > radius:
        return {"label":"顺势收益型","icon":"➡️",
                "definition":"业绩重心高度集中于高财力区域，转化效率接近均值水平，其产出与区域资源禀赋强相关，尚未形成独立于环境之外的差异化竞争能力。",
                "management":["重点监控其负责区域的财政健康度，一旦区域出现下行，需提前介入而非等待业绩滑落","创造跨环境展业机会，观察脱离资源红利后的真实能力水位"],
                "color":"#82E0AA"}
    elif abs(y) <= radius and x < -radius:
        return {"label":"逆境生存型","icon":"⬅️",
                "definition":"业绩重心集中于低财力区域，在资源匮乏的环境下维持接近均值的转化效率，具备一定的逆境作战韧性，但尚未形成非对称的竞争优势。",
                "management":["适合派往竞争对手渗透薄弱、资源条件较差的市场做初期开拓","与开疆拓土型人才的区别在于效率，需重点观察其在同类艰苦区域的效率提升空间",'注意心理状态管理，长期深耕艰苦区域易产生倦怠，需配合激励机制'],
                "color":"#5DADE2"}
    # 四象限复合区
    if x < 0 and y > 0:
        return {"label":"开疆拓土型","icon":"🔵",
                "definition":"主动选择低财力省份展业，却实现全司高位产出，具备极强的逆势开拓能力、强烈的自驱力和非对称竞争优势。",
                "management":["优先复盘其在艰苦区域的具体打法，提炼可迁移的竞争策略作为新市场开拓样板","是开辟新区域、对抗强竞对的首选人选，给予充分的自主空间和资源配置权",'警惕将其长期困于艰苦区域导致流失，适时给予更大舞台'],
                "color":"#1565C0"}
    elif x > 0 and y > 0:
        return {"label":"核心引擎型","icon":"🟢",
                "definition":"在高财力省份实现高产出，是公司核心基本盘的主要守护者，兼具资源整合能力与高效执行力。",
                "management":["优先保障其负责区域的资源稳定性，避免因内部调整影响基本盘","定期做横向对比评估，确认高业绩来自个人能力而非单纯区域红利，为晋升决策提供依据"],
                "color":"#2E7D32"}
    elif x > 0 and y < 0:
        return {"label":"资源错配型","icon":"🔴",
                "definition":'业绩重心集中于高财力区域，但转化效率持续低于均值，呈现"占据优质资源却未能有效转化"的状态，是资源配置效率最低的人员类型。',
                "management":["立即排查获客、跟进、促成各环节，重点核查是否存在客户资源虚占、过程动作缺失等问题",'评估人岗匹配度：区分"能力不足"与"动力不足"，前者需辅导，后者需激励或警示',
                '设定短期改善目标（建议1个季度），若无明显改善则考虑区域资源重新分配','此类人员是管理资源投入优先级很高的对象'],
                "color":"#E74C3C"}
    else:
        return {"label":"低位蛰伏型","icon":"🟤",
                "definition":"业绩重心集中于低财力区域，转化效率低于均值，在资源有限的环境下维持低频产出，整体处于低位运行状态。",
                "management":["若为新人：重点看学习速度和主动性，给予明确的短期目标和反馈机制","老员工：评估是否已触及个人天花板，若连续多期无改善则考虑岗位调整",'管理精力应聚焦在有改善潜力的个体，避免平均分配注意力'],
                "color":"#6D4C41"}

st.markdown(f"""
<style>
  .block-container{{padding-top:3rem!important;padding-bottom:1rem!important}}
  html,body,[class*="css"]{{font-family:{FONT}}}
  .stApp{{background:#FDFCFB}}
  section[data-testid="stSidebar"]{{background:#F5F2F0}}
  [data-testid="metric-container"]{{background:#F5F2F0;border:1px solid #E8E4E0;border-radius:10px;padding:12px 16px}}
  [data-testid="metric-container"] label{{color:#6B6560;font-size:.76rem}}
  [data-testid="metric-container"] [data-testid="metric-value"]{{color:#1F1F1F;font-size:1.3rem;font-weight:600}}
  .sec-title{{display:inline-block;font-size:1.08rem;font-weight:700;color:{C_TITLE};
    padding-bottom:.5rem;border-bottom:2px solid {C_TITLE};margin-bottom:1rem;margin-top:1.2rem}}
  .hero-card{{background:#F5F2F0;border-radius:8px;padding:10px 14px;margin-bottom:6px;border:1px solid #E8E4E0}}
  .hero-name{{font-size:1rem;font-weight:600;color:#1F1F1F}}
  .hero-val{{font-size:.84rem;color:#607D8B;margin-top:2px}}
  .note-teal{{background:#E8F5F4;border:1px solid {C_TEAL};border-radius:8px;padding:10px 16px;font-size:.82rem;color:#2C5555;margin-bottom:10px}}
  .caption-box{{background:#F5F2F0;border:1px solid #E8E4E0;border-radius:8px;padding:8px 14px;font-size:.76rem;color:#9E9690;margin-top:14px}}
  .type-card{{border-radius:8px;padding:12px 16px;margin-bottom:8px;border-left:4px solid;background:#FAFAF8}}
  .type-card h4{{margin:0 0 4px;font-size:.9rem}} .type-card p{{margin:0;font-size:.76rem;color:#555;line-height:1.5}}
  /* rank scroll container */
  .rank-scroll{{height:220px;overflow-y:auto;padding-right:6px;
    mask-image:linear-gradient(to bottom,black 80%,transparent 100%);
    -webkit-mask-image:linear-gradient(to bottom,black 80%,transparent 100%)}}
  /* metric mini */
  .metric-mini{{background:#F5F2F0;border:1px solid #E8E4E0;border-radius:8px;padding:10px 12px;margin-bottom:8px}}
  .metric-mini .lbl{{font-size:.72rem;color:#6B6560}}
  .metric-mini .val{{font-size:1.15rem;font-weight:600;color:#1F1F1F}}
  .stTabs [data-baseweb="tab"]{{font-size:.88rem;padding:7px 16px}}
  .stTabs [aria-selected="true"]{{color:{C_TITLE};font-weight:600;border-bottom:2px solid {C_TITLE}}}
</style>
""", unsafe_allow_html=True)

def sec(title):
    st.markdown(f'<div class="sec-title">{title}</div>', unsafe_allow_html=True)

PROVINCE_COORDS = {
    "北京":(116.41,39.92),"天津":(117.19,39.13),"河北":(114.53,38.04),
    "山西":(112.55,37.87),"内蒙古":(111.75,40.84),"辽宁":(123.43,41.80),
    "吉林":(125.33,43.90),"黑龙江":(126.66,45.75),"上海":(121.47,31.23),
    "江苏":(118.76,32.06),"浙江":(120.15,30.27),"安徽":(117.28,31.86),
    "福建":(119.30,26.08),"江西":(115.90,28.68),"山东":(117.00,36.67),
    "河南":(113.65,34.76),"湖北":(114.30,30.60),"湖南":(112.98,28.20),
    "广东":(113.26,23.13),"广西":(108.32,22.82),"海南":(110.35,20.02),
    "重庆":(106.55,29.56),"四川":(104.07,30.67),"贵州":(106.71,26.60),
    "云南":(102.71,25.05),"西藏":(91.13,29.66),"陕西":(108.95,34.27),
    "甘肃":(103.83,36.06),"青海":(101.78,36.62),"宁夏":(106.27,38.47),
    "新疆":(87.62,43.79),
}

DATA_DIR = Path(__file__).parent /"data"


@st.cache_data
def load_arrivals():
    f = DATA_DIR/"全司到款明细（2021-2022）.xlsx"
    d21=pd.read_excel(f,sheet_name="2021年"); d21.columns=["省份","业绩(万元)","市场人员"]; d21["年度"]="2021年"
    d22=pd.read_excel(f,sheet_name="2022年"); d22.columns=["省份","业绩(万元)","市场人员"]; d22["年度"]="2022年"
    df=pd.concat([d21,d22],ignore_index=True).dropna(subset=["省份","业绩(万元)","市场人员"])
    df["业绩(万元)"]=pd.to_numeric(df["业绩(万元)"],errors="coerce")
   
    # 中国省份名称中，除了"黑龙江"和"内蒙古"是 3 字简称外，其余均为 2 字。
    # 如果是这两个省，取前 3 位，否则取前 2 位。
    df["省份"] = df["省份"].apply(lambda x: x[:3] if x[:3] in ["内蒙古", "黑龙江"] else x[:2])
    
    return df.dropna(subset=["业绩(万元)"])

@st.cache_data
def load_fiscal():
    # 1. 让内部函数接收 sheet 参数
    def _r(p, sheet, yr):
        # 2. 将 sheet 传给 pd.read_excel
        df = pd.read_excel(p, sheet_name=sheet) 
        df.columns = ["省份", "财力(亿元)", "债券发行数量(只)"] 
        
        df["省份"] = df["省份"].apply(
            lambda x: x[:3] if str(x)[:3] in ["内蒙古", "黑龙江"] else str(x)[:2]
        ) 
        df["年度"] = yr
        return df
        
    # 3. 严格按照 (路径, 工作表名, 年份) 的顺序传参
    return pd.concat([
        _r(DATA_DIR / "全国各省财力及债券数据（2021-2022）.xlsx", "2021年", "2021年"),
        _r(DATA_DIR / "全国各省财力及债券数据（2021-2022）.xlsx", "2022年", "2022年")
    ], ignore_index=True) # 建议加上 ignore_index=True 充置索引



@st.cache_data
def load_dept_revenue():
    df=pd.read_excel(DATA_DIR/"业务一部收入明细（2023年1-10月）.xlsx")
    df["金额"]=pd.to_numeric(df["金额"],errors="coerce")
    return df.dropna(subset=["金额","市场人员"])


@st.cache_data
def load_dept_expense():
    bx=pd.read_excel(DATA_DIR/"业务一部报销明细（2023年1-10月）.xlsx")
    bx["金额"]=pd.to_numeric(bx["金额"],errors="coerce")
    bx=bx.dropna(subset=["金额","申请人"])
    bx["日期"]=pd.to_datetime(bx["申请日期"],errors="coerce").dt.strftime("%Y-%m-%d")
    bx["费用大类"]=bx["科目"].apply(lambda x:"招待费" if str(x) in ["业务接待费","招待费"] else "差旅费")
    bx["市场人员"]=bx["申请人"]; bx=bx[["日期","市场人员","金额","费用大类","科目"]]
    xc=pd.read_excel(DATA_DIR/"业务一部携程商旅明细（2023年1-10月）.xlsx")
    xc.columns=["部门","科目","申请人","日期1","日期2","金额"]
    xc=xc.dropna(subset=["申请人"]); xc["科目"]=xc["科目"].ffill()
    xc["金额"]=pd.to_numeric(xc["金额"],errors="coerce"); xc=xc.dropna(subset=["金额"])
    xc["日期1s"]=pd.to_datetime(xc["日期1"],errors="coerce").dt.strftime("%Y-%m-%d")
    xc["日期2s"]=pd.to_datetime(xc["日期2"],errors="coerce").dt.strftime("%Y-%m-%d")
    xc["日期"]=xc.apply(lambda r:f"{r['日期1s']}至{r['日期2s']}" if pd.notna(r["日期2"]) else r["日期1s"],axis=1)
    xc["费用大类"]="差旅费"; xc["市场人员"]=xc["申请人"]; xc=xc[["日期","市场人员","金额","费用大类","科目"]]
    return pd.concat([bx,xc],ignore_index=True)

def compute_stats(df,year):
    sub=df[df["年度"]==year]
    g=sub.groupby(["省份","市场人员"]).agg(业绩总额=("业绩(万元)","sum"),成单数=("业绩(万元)","count")).reset_index()
    g["单笔均额(万元)"]=(g["业绩总额"]/g["成单数"]).round(2); return g

def assign_tier(prov_df):
    prov_df = prov_df.copy()
    
    # 1. 筛选出有业绩的数据
    active_mask = prov_df["业绩总额"] > 0
    active = prov_df[active_mask]
    
    # 2. 如果没有活跃数据，直接处理特殊情况
    if active.empty:
        prov_df["档位"] = "无业绩"
        return prov_df
        
    # 3. 使用 pandas 完美的 qcut 功能，直接把数据按"业绩总额"降序分成 3 组
    # labels 对应你的：前1/3 (业绩高), 中1/3, 后1/3 (业绩低)
    # 因为要降序，我们可以对"业绩总额"取负号，或者直接切分后反转标签
    try:
        prov_df.loc[active_mask, "档位"] = pd.qcut(
            active["业绩总额"], 
            q=3, 
            labels=["后1/3", "中1/3", "前1/3"]
        ).astype(str)
    except ValueError:
        # 防止数据重复值过多导致 qcut 报错，加入 duplicates='drop'
        prov_df.loc[active_mask, "档位"] = pd.qcut(
            active["业绩总额"], 
            q=3, 
            labels=["后1/3", "中1/3", "前1/3"],
            duplicates='drop'
        ).astype(str)
        
    # 4. 把业绩为 0 的填补为 "无业绩"
    prov_df["档位"] = prov_df["档位"].fillna("无业绩")
    
    return prov_df
def cluster_provinces(df_stats,n):
    prov=df_stats.groupby("省份").agg(业绩总额=("业绩总额","sum"),成单数=("成单数","sum")).reset_index()
    prov["单笔均额(万元)"]=prov["业绩总额"]/prov["成单数"]
    X=StandardScaler().fit_transform(prov[["单笔均额(万元)","成单数"]])
    km=KMeans(n_clusters=n,random_state=42,n_init=10); prov["cluster"]=km.fit_predict(X)
    centers=pd.DataFrame(km.cluster_centers_,columns=["均额","成单"])
    hp=centers["均额"].median(); hc=centers["成单"].median()
    lmap={}
    for i,r in centers.iterrows():
        a,b=r["均额"]>=hp,r["成单"]>=hc
        lmap[i]="大客户维护型" if a and not b else("散户扩张型" if not a and b else("优质核心市场" if a and b else "边缘待评估市场"))
    prov["市场类型"]=prov["cluster"].map(lmap); return prov

def top_scope(df_stats,province="全部",n=5):
    g=df_stats.groupby("市场人员")["业绩总额"].sum().reset_index() if province=="全部" \
        else df_stats[df_stats["省份"]==province].groupby("市场人员")["业绩总额"].sum().reset_index()
    return g.sort_values("业绩总额",ascending=False).head(n).reset_index(drop=True)

# 图表公共layout参数（legend右外侧，toolbox右上角高于legend）
def chart_layout(height=480, legend_y=0.7, extra=None):
    d = dict(
        height=height, plot_bgcolor="#FDFCFB", paper_bgcolor="#FDFCFB",
        font=dict(family=FONT, size=12),
        margin=dict(l=10, r=160, t=40, b=10),
        legend=dict(
            orientation="v", x=1.02, y=legend_y,
            bgcolor="rgba(255,255,255,.92)", bordercolor="#E0E0E0", borderwidth=1,
            itemsizing="constant",
        ),
    )
    if extra:
        d.update(extra)
    return d

try:
    df_arr=load_arrivals(); df_fis=load_fiscal()
    df_rev=load_dept_revenue(); df_exp=load_dept_expense()
except FileNotFoundError as e:
    st.error(f"数据文件未找到：{e}"); st.stop()


with st.sidebar:
    st.markdown(f"<h3 style='color:{C_TITLE}'>市场战略及资源配置分析平台</h3>",unsafe_allow_html=True)
    st.markdown('<p style="font-size:.72rem;color:#9E9690;text-transform:uppercase;letter-spacing:.06em;margin:14px 0 5px">切换视角</p>',unsafe_allow_html=True)
    module=st.selectbox("模块",["全司业务概览","部门人效分析"],label_visibility="collapsed")
    st.divider()
    st.caption("数据说明：\n\n - 公司数据均为参照真实业务场景构造的虚拟数据，仅用于展示本项目的分析逻辑。\n\n- 财力数据为地方全级一般公共预算收入。\n\n- 债券数据为该省该年的城投债发行数量。")



@st.cache_data  # 💡 必须加缓存，否则每次刷新页面都会重新下载 1MB 的地图数据，速度会慢
def get_china_map_data():
    """
    同时获取省界（带清洗）和全国轮廓（不带省界）
    优先从远程获取，成功则缓存到本地；远程失败则读取本地缓存
    """
    import json
    CACHE_DIR = Path(__file__).parent / "data"
    PROV_CACHE = CACHE_DIR / "china_provinces_geo.json"
    COUNTRY_CACHE = CACHE_DIR / "china_country_geo.json"

    prov_url = "https://geo.datav.aliyun.com/areas_v3/bound/100000_full.json"
    country_url = "https://geo.datav.aliyun.com/areas_v3/bound/100000.json"

    def _fetch_json(url, timeout=20):
        """带超时的 GET 请求，返回 JSON"""
        return requests.get(url, timeout=timeout, headers={
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64)"
        })

    prov_geojson = None
    country_geojson = None

    # ── 1. 省份数据 ──
    try:
        res_p = _fetch_json(prov_url)
        prov_geojson = res_p.json()
        # 清洗省份名称
        for feature in prov_geojson.get("features", []):
            full_name = feature["properties"]["name"]
            short_name = (full_name.replace("省", "")
                                   .replace("市", "")
                                   .replace("特别行政区", "")
                                   .replace("自治区", "")
                                   .replace("壮族", "")
                                   .replace("回族", "")
                                   .replace("维吾尔", ""))
            feature["properties"]["name"] = short_name
        # 缓存到本地
        with open(PROV_CACHE, "w", encoding="utf-8") as f:
            json.dump(prov_geojson, f, ensure_ascii=False)
    except Exception as e:
        st.warning(f"省份地图远程加载失败（{e}），尝试本地缓存…")
        if PROV_CACHE.exists():
            with open(PROV_CACHE, "r", encoding="utf-8") as f:
                prov_geojson = json.load(f)
            st.info("已从本地缓存加载省份地图")

    # ── 2. 全国轮廓 ──
    try:
        res_c = _fetch_json(country_url)
        country_geojson = res_c.json()
        with open(COUNTRY_CACHE, "w", encoding="utf-8") as f:
            json.dump(country_geojson, f, ensure_ascii=False)
    except Exception as e:
        st.warning(f"全国轮廓远程加载失败（{e}），尝试本地缓存…")
        if COUNTRY_CACHE.exists():
            with open(COUNTRY_CACHE, "r", encoding="utf-8") as f:
                country_geojson = json.load(f)
            st.info("已从本地缓存加载全国轮廓")

    # ── 3. 最终判断 ──
    if prov_geojson is None or country_geojson is None:
        st.error("地图数据加载失败：远程不可达且无本地缓存。请检查网络后刷新页面。")
        return None, None
    return prov_geojson, country_geojson

#  获取处理后的地图 同时接收两个变量
province_geojson, country_geojson = get_china_map_data()

# ═══════════════════════════════════════════════════════
# 模块一：全司业务概览
# ═══════════════════════════════════════════════════════
if module=="全司业务概览":
    st.markdown(f"<h2 style='color:{C_TITLE};margin-bottom:4px'>全司业务概览</h2>",unsafe_allow_html=True)

    year_sel=st.selectbox("选择年度",["2021年","2022年"],key="year_sel")
    df_year=df_arr[df_arr["年度"]==year_sel]
    df_stats=compute_stats(df_arr,year_sel)
    df_fis_y=df_fis[df_fis["年度"]==year_sel].copy()
    df_fis_y["财力排名"]=df_fis_y["财力(亿元)"].rank(ascending=False,method="min").astype(int)
    df_fis_y["发行排名"]=df_fis_y["债券发行数量(只)"].rank(ascending=False,method="min").astype("Int64")

    # ── 市场版图
    sec("市场版图")
    prov_perf=df_year.groupby("省份").agg(业绩总额=("业绩(万元)","sum"),成单总数=("业绩(万元)","count")).reset_index()
    prov_perf["单笔均额"]=(prov_perf["业绩总额"]/prov_perf["成单总数"]).round(1)
    prov_perf["业绩排名"]=prov_perf["业绩总额"].rank(ascending=False,method="min").astype(int)

    person_d=(df_year.groupby(["省份","市场人员"])["业绩(万元)"].sum()
              .reset_index().sort_values(["省份","业绩(万元)"],ascending=[True,False]))
    pstr=person_d.groupby("省份").apply(
        lambda d:"<br>".join(f"  {r['市场人员']} {r['业绩(万元)']:.1f}万" for _,r in d.iterrows())
    ).reset_index(); pstr.columns=["省份","人员明细"]
    prov_perf=prov_perf.merge(pstr,on="省份",how="left"); prov_perf["人员明细"]=prov_perf["人员明细"].fillna("暂无")

    all_prov=pd.DataFrame([(p,lon,lat) for p,(lon,lat) in PROVINCE_COORDS.items()],columns=["省份","lon","lat"])
    prov_full=all_prov.merge(prov_perf,on="省份",how="left")
    prov_full["业绩总额"]=prov_full["业绩总额"].fillna(0)
    prov_full["成单总数"]=prov_full["成单总数"].fillna(0).astype(int)
    prov_full["单笔均额"]=prov_full["单笔均额"].fillna(0)
   
    prov_full["人员明细"]=prov_full["人员明细"].fillna("暂无业绩")
    prov_full=assign_tier(prov_full)

    top5_provs=prov_full.sort_values("业绩总额",ascending=False).head(5)["省份"].tolist()
    rank_labels=["No.1","No.2","No.3","No.4","No.5"]
    top5_map={p:rank_labels[i] for i,p in enumerate(top5_provs)}
    rank_emoji={"No.1":"🥇","No.2":"🥈","No.3":"🥉","No.4":"","No.5":""}

    fig_map=go.Figure()
    # --- 图层 1：省界（灰色细线） ---
    if province_geojson:
        fig_map.add_trace(go.Choropleth(
            geojson=province_geojson,
            locations=prov_full["省份"],
            z=[0] * len(prov_full),
            featureidkey="properties.name",
            colorscale=[[0, 'rgba(0,0,0,0)'], [1, 'rgba(0,0,0,0)']], # 透明填充
            showscale=False,
            marker=dict(line=dict(color="#D9D9D9", width=0.8)), # 灰色细线
            hoverinfo="skip"
        ))

    # --- 图层 2：中国全境轮廓（灰色细线） ---
    if country_geojson:
        fig_map.add_trace(go.Choropleth(
            geojson=country_geojson,
            locations=[country_geojson['features'][0]['properties']['name']], # 匹配 API 里的 "中国"
            z=[0],
            featureidkey="properties.name",
            colorscale=[[0, 'rgba(0,0,0,0)'], [1, 'rgba(0,0,0,0)']], # 透明填充
            showscale=False,
            marker=dict(line=dict(color="#CCCCCC", width=1.5)), # 黑色粗线
            hoverinfo="skip"
        ))
    mv=max(prov_full["业绩总额"].max(),1)
    for tier,color in TIER_COLORS.items():
        sub=prov_full[prov_full["档位"]==tier]
        if sub.empty: continue
        sz=(sub["业绩总额"]/mv*40+10).clip(lower=10) if tier!="无业绩" else pd.Series([10]*len(sub),index=sub.index)
        ht=sub.apply(lambda r:(
            f"<b>{r['省份']}</b><br>"
            f"业绩排名：全国第{r['业绩排名']:.0f}<br>"
            f"总业绩：{r['业绩总额']:.1f}万元<br>"
            f"总单数：{r['成单总数']}笔<br>"
            f"每单均额：{r['单笔均额']:.1f}万元<br>------<br>{r['人员明细']}"),axis=1)
        # 所有省份名放圆圈外上方
        fig_map.add_trace(go.Scattergeo(
            lon=sub["lon"],lat=sub["lat"],
            text=sub["省份"],hovertext=ht,hoverinfo="text",
            mode="markers+text",textposition="top center",
            textfont=dict(size=11,color="#444"),
            marker=dict(size=sz,color=color,opacity=.85,line=dict(color="white",width=.8)),
            name=tier,showlegend=True))
        # Top5省份在圆圈内额外叠加No.X标注（银灰色#EFEFEF粗体居中，字体放大）
        top5_sub=sub[sub["省份"].isin(top5_map.keys())]
        if not top5_sub.empty:
            fig_map.add_trace(go.Scattergeo(
                lon=top5_sub["lon"],lat=top5_sub["lat"],
                text=top5_sub["省份"].map(top5_map),hoverinfo="skip",
                mode="text",textposition="middle center",
                textfont=dict(size=11,color="#EFEFEF",family="Arial Black"),
                showlegend=False))
    fig_map.update_layout(
        geo=dict(
            visible=False,          # 隐藏原生底图
            showland=True,          # 开启陆地显示
            landcolor="#F8F6F2",    # 陆地底色
            showcountries=False,     # 既然用了自定义 GeoJSON，原生的国家线可以关掉，避免重叠
            showsubunits=True,      # 必须为 True，显示省界
            subunitcolor="#CCCCCC", 
            center=dict(lon=105, lat=36), # 修改 1：锁定中国中心经纬度
            projection_scale=5,           # 修改 2：放大倍数（从原来的 3.5 改为 6）
            # ... 其他保持不变
        ),
        legend=dict(title=dict(text="业绩档位"),orientation="v",x=0.01,y=0.98,
                    bgcolor="rgba(255,255,255,.88)",bordercolor="#E0E0E0",borderwidth=1,itemsizing="constant"),
        margin=dict(l=0,r=0,t=0,b=0),height=500,paper_bgcolor="#FDFCFB")


    
    st.plotly_chart(fig_map,use_container_width=True)
    st.caption("气泡大小=业绩规模 | 绿=前1/3  黄=中1/3  红=后1/3  灰=无业绩 | 圆圈内Top5标No.1-5 | 悬停查看详情")

    # ── 省份业绩排名榜 + 战神榜 并列等高
    col_rank, col_hero = st.columns(2)

    with col_rank:
        sec("省份业绩排名榜")
        prov_rank_all=(df_year.groupby("省份").agg(业绩总额=("业绩(万元)","sum"),成单总数=("业绩(万元)","count"))
                       .reset_index().sort_values("业绩总额",ascending=False).reset_index(drop=True))
        prov_rank_all["单笔均额"]=(prov_rank_all["业绩总额"]/prov_rank_all["成单总数"]).round(2)

        RANK_EMOJI=["🥇","🥈","🥉"]
        cards_html=""
        for i,(_,row) in enumerate(prov_rank_all.iterrows()):
            rank=i+1
            emoji=RANK_EMOJI[i] if i<3 else ""
            persons2=(df_year[df_year["省份"]==row["省份"]].groupby("市场人员")["业绩(万元)"].sum()
                      .sort_values(ascending=False))
            ptags="　".join(f"{p} {v:.1f}万" for p,v in persons2.items())
            cards_html+=(
                f"<div style='background:#F5F2F0;border-radius:8px;padding:9px 13px;margin-bottom:5px;border:1px solid #E8E4E0'>"
                f"<span style='font-weight:700;color:#1F1F1F'>No.{rank} {emoji}</span>&nbsp;&nbsp;"
                f"<span style='font-size:1rem;font-weight:600;color:#1F1F1F'>{row['省份']}</span>"
                f"<div style='font-size:.84rem;color:#607D8B;margin-top:2px'>总业绩 {row['业绩总额']:.1f}万 · 成单 {row['成单总数']}笔 · 单笔均额 {row['单笔均额']}万</div>"
                f"<div style='font-size:.78rem;color:#888;margin-top:2px'>{ptags}</div></div>")
        # 滚动容器，仅露出前5条（约480px）
        st.markdown(f'<div class="rank-scroll" style="height:490px">{cards_html}</div>',unsafe_allow_html=True)

    with col_hero:
        sec("战神榜")
        prov_sorted_hero=(df_year.groupby("省份")["业绩(万元)"].sum()
                          .reset_index().sort_values("业绩(万元)",ascending=False))
        prov_opts_hero=["全部"]+[f"{r['省份']}（{r['业绩(万元)']:.0f}万元）" for _,r in prov_sorted_hero.iterrows()]
        prov_raw_opts=["全部"]+prov_sorted_hero["省份"].tolist()
        col_filter_h,_=st.columns([3,1])
        with col_filter_h:
            prov_idx=st.selectbox("筛选省份（按业绩降序）",range(len(prov_opts_hero)),
                                  format_func=lambda i:prov_opts_hero[i],key="hero_prov_idx")
        prov_filter=prov_raw_opts[prov_idx]

        ds=compute_stats(df_arr,year_sel)
        #top5h=top_scope(ds,prov_filter,n=5)
        top5h = top_scope(ds, prov_filter, n=len(ds))#不止要top5，要全部展示，滚动容器
        hero_html=""
        if top5h.empty:
            hero_html="<div style='color:#888;padding:20px'>该省份暂无数据</div>"
        else:
            for idx,row in top5h.iterrows():
                cnt=ds[ds["市场人员"]==row["市场人员"]]["成单数"].sum()
                avg=row["业绩总额"]/cnt if cnt>0 else 0
                rank_n=idx+1; emoji=RANK_EMOJI[idx] if idx<3 else ""
                hero_html+=(
                    f"<div style='background:#F5F2F0;border-radius:8px;padding:9px 13px;margin-bottom:5px;border:1px solid #E8E4E0'>"
                    f"<span style='font-weight:700;color:#1F1F1F'>No.{rank_n} {emoji}</span>&nbsp;"
                    f"<span style='font-size:1rem;font-weight:600;color:#1F1F1F'>{row['市场人员']}</span>"
                    f"<div style='font-size:.84rem;color:#607D8B;margin-top:2px'>业绩 {row['业绩总额']:.1f}万 · 成单 {cnt:.0f}笔 · 单笔均额 {avg:.1f}万</div>"
                    f"</div>")
        # 滚动容器，露出前5名（约390px）
        st.markdown(f'<div class="rank-scroll" style="height:390px">{hero_html}</div>',unsafe_allow_html=True)

    # ── 省域财力与业绩分析·攻守战略
    sec("省域财力与业绩分析·攻守战略")
    st.markdown(
        "<div style='font-size:.84rem;color:#555;line-height:1.8;margin-bottom:10px'>"
       
        "本模块通过地方财力与业绩收入的四象限分析，锚定各省份的战略属性，帮助管理层动态优化兵力部署与资源倾斜。<br>"
        "<b>地方财力：</b>代表这个省的势能，是地方市场自带的能量。&emsp;"
        "<b>公司业绩：</b>代表这个省的效能，即市场人员转化了多少能量。<br>"
        "<b>X轴</b>：省份一般公共预算收入（亿元）；&emsp;"
        "<b>Y轴</b>：全司省份总业绩额（万元）；&emsp;"
        "<b>气泡大小</b>：当年债券发行数量（只）。<br>"
        "</div>",unsafe_allow_html=True)


    POTENCY_LABELS=[
        ("深耕奇迹区",C_QB,"左上 🔵","低财力·高业绩","财力偏弱意味着市场势能天然受限，在此逆境下依然跑赢半数省份。气泡大→市场融资动能强但财力相对偏低，属于城投活跃型弱财力省份，是真正的逆势机会窗口，需加码人员配置；气泡小→融资体量受限，市场天花板清晰，应以维护存量为主。"),
        ("成熟粮仓区",C_QG,"右上 🟢","高财力·高业绩","财力与业绩双强，势能与效能共振，是公司基本盘与利润来源。气泡大→市场处于融资活跃期，增量窗口开启，需主动防守竞对切入，同时扩大覆盖密度；气泡小→市场趋于沉淀，应降低边际成本、强化关系维护，从抢增量切换为守存量模式。"),
        ("边缘维持区",C_QBROWN,"左下 🟤","低财力·低业绩","财力偏弱的环境下业绩跑输半数省份，适合轻资产运营。气泡大→出现结构性矛盾：市场融资动能客观存在，但公司颗粒无收，需排查是人员缺位、准入障碍还是竞对垄断；气泡小→市场本身低迷，维持基础覆盖即可，避免过度投入，以最低成本保留战略选择权。"),
        ("高潜低收区",C_QRED,"右下 🔴","高财力·低业绩","财力强劲但业绩落后，势能充裕却创收乏力，是公司战略诊断的第一优先级。气泡大→融资市场高度活跃而我们缺席，属于严重的机会错失，需排查是否存在资质短板、商务关系被竞对垄断、人员配置等。气泡小→市场动能虽弱但财力基础仍在，属于潜伏型机会区，可提前布局关系网络，等待融资周期回暖时快速转化。"),
    ]
    pot_cols=st.columns(4)
    for (lname,color,pos,sub,desc),col in zip(POTENCY_LABELS,pot_cols):
        col.markdown(
            f"<div class='type-card' style='border-left-color:{color}'>"
            f"<h4 style='color:{color}'>{lname} <span style='font-size:.7rem;font-weight:400'>{pos}</span></h4>"
            f"<p style='color:#777;font-size:.72rem;margin-bottom:3px'>{sub}</p>"
            f"<p style='font-size:.76rem'>{desc}</p></div>",unsafe_allow_html=True)

    # 构建散点数据
    prov_perf2=df_year.groupby("省份").agg(业绩总额=("业绩(万元)","sum"),成单总数=("业绩(万元)","count")).reset_index()
    prov_perf2["业绩排名_省"]=prov_perf2["业绩总额"].rank(ascending=False,method="min").astype(int)
    pd_detail=(df_year.groupby(["省份","市场人员"])
               .agg(业绩=("业绩(万元)","sum"),单数=("业绩(万元)","count"))
               .reset_index().sort_values(["省份","业绩"],ascending=[True,False]))
    pd_detail["均额"]=(pd_detail["业绩"]/pd_detail["单数"]).round(1)
    def build_pstr(prov):
        rows=pd_detail[pd_detail["省份"]==prov]
        return "<br>".join(f"  {r['市场人员']} {r['业绩']:.1f}万 {int(r['单数'])}单 每单均额{r['均额']}万" for _,r in rows.iterrows()) if len(rows) else "暂无"

    all_pf=pd.DataFrame([(p,lon,lat) for p,(lon,lat) in PROVINCE_COORDS.items()],columns=["省份","lon","lat"])
    scatter_df=all_pf.merge(df_fis_y[["省份","财力(亿元)","债券发行数量(只)","财力排名","发行排名"]],on="省份",how="left")
    scatter_df=scatter_df.merge(prov_perf2[["省份","业绩总额","成单总数","业绩排名_省"]],on="省份",how="left")
    scatter_df["业绩总额"]=scatter_df["业绩总额"].fillna(0)
    scatter_df["成单总数"]=scatter_df["成单总数"].fillna(0).astype(int)
    scatter_df["业绩排名_省"]=scatter_df["业绩排名_省"].fillna(99).astype(int)
    scatter_df["债券发行数量(只)"]=scatter_df["债券发行数量(只)"].fillna(scatter_df["债券发行数量(只)"].median())

    scatter_df["人员明细"]=scatter_df["省份"].apply(build_pstr)

    # ── 市场人员的综合战场偏好指数：预计算百分位参照系 ──
    # X轴：各省财力在全国的百分位 → 连续值，保留程度信息
    _fiscal_rank = scatter_df[["省份","财力(亿元)"]].dropna(subset=["财力(亿元)"])
    fiscal_pct_map = _fiscal_rank.set_index("省份")["财力(亿元)"].rank(pct=True).to_dict()
    # Y轴：全司所有人员的总业绩分布，用于衡量个人绝对业绩在全公司的排位
    #      （全国展业前提下，市场选择是个人能力的一部分，Y轴只用绝对产出）
    all_person_totals = df_stats.groupby("市场人员")["业绩总额"].sum().values
    CENTROID_SCALE = 2.0  # 放大重心偏离度，避免人员过度聚集在原点

    fis_mid_s=scatter_df["财力(亿元)"].dropna().median()
    perf_avg_s=scatter_df[scatter_df["业绩总额"]>0]["业绩总额"].median()
    x_max_s=scatter_df["财力(亿元)"].dropna().max()*1.18
    y_max_s=scatter_df["业绩总额"].max()*1.3

    def pot_quad(row):
        if pd.isna(row["财力(亿元)"]): return "边缘维持区"
        hi_x=row["财力(亿元)"]>=fis_mid_s; hi_y=row["业绩总额"]>=perf_avg_s
        return "成熟粮仓区" if hi_x and hi_y else("深耕奇迹区" if not hi_x and hi_y else("高潜低收区" if hi_x else "边缘维持区"))
    scatter_df["象限"]=scatter_df.apply(pot_quad,axis=1)
    max_iss=scatter_df["债券发行数量(只)"].max()
    scatter_df["气泡大小"]=scatter_df["债券发行数量(只)"]/max_iss*45+8
    scatter_df["hover"]=scatter_df.apply(lambda r:(
        f"<b>{r['省份']}【{r['象限']}】</b><br>"
        f"地方财力：{r['财力(亿元)']:.0f}亿元（全国排名第{r['财力排名']:.0f}）<br>"
        f"当年发行：{int(r['债券发行数量(只)'])}只（全国排名第{r['发行排名']:.0f}）<br>"
        f"全司业绩：{r['业绩总额']:.1f}万元<br>"
        f"省份业绩排名：全司第{r['业绩排名_省']}<br>"
        f"---<br>业绩贡献人员：<br>{r['人员明细']}"),axis=1)

    fig_pot=go.Figure()
    POT_BG=[("rgba(21,101,192,.06)",0,fis_mid_s,perf_avg_s,y_max_s),
            ("rgba(46,125,50,.06)", fis_mid_s,x_max_s,perf_avg_s,y_max_s),
            ("rgba(109,76,65,.06)", 0,fis_mid_s,0,perf_avg_s),
            ("rgba(198,40,40,.06)", fis_mid_s,x_max_s,0,perf_avg_s)]
    POT_ANN=[("深耕奇迹区",fis_mid_s*.3,(perf_avg_s+y_max_s)/2),
             ("成熟粮仓区",(fis_mid_s+x_max_s)/2,(perf_avg_s+y_max_s)/2),
             ("边缘维持区",fis_mid_s*.3,perf_avg_s*.3),
             ("高潜低收区",(fis_mid_s+x_max_s)/2,perf_avg_s*.3)]
    for fill,x0,x1,y0,y1 in POT_BG:
        fig_pot.add_shape(type="rect",x0=x0,x1=x1,y0=y0,y1=y1,fillcolor=fill,line_width=0,layer="below")
    for lbl,lx,ly in POT_ANN:
        fig_pot.add_annotation(x=lx,y=ly,text=lbl,showarrow=False,font=dict(size=11,color="rgba(0,0,0,.18)"),xanchor="center")
    fig_pot.add_hline(y=perf_avg_s,line_dash="dash",line_color="#90A4AE",line_width=1,
                      annotation_text=f"省业绩中位 {perf_avg_s:.0f}万",annotation_position="right")
    fig_pot.add_vline(x=fis_mid_s,line_dash="dash",line_color="#90A4AE",line_width=1,
                      annotation_text=f"财力中位 {fis_mid_s:.0f}亿",annotation_position="top")
    for quad in ["深耕奇迹区","成熟粮仓区","边缘维持区","高潜低收区"]:
        sub=scatter_df[scatter_df["象限"]==quad].dropna(subset=["财力(亿元)"])
        if sub.empty: continue
        fig_pot.add_trace(go.Scatter(
            x=sub["财力(亿元)"],y=sub["业绩总额"],mode="markers+text",
            text=sub["省份"],textposition="top center",textfont=dict(size=9),
            marker=dict(size=sub["气泡大小"],color=POTENCY_COLORS[quad],opacity=.82,line=dict(color="white",width=1)),
            name=quad,hovertext=sub["hover"],hoverinfo="text"))
    fig_pot.update_layout(**chart_layout(520, 0.98, {
        "xaxis_title":"省份一般公共预算收入（亿元）","yaxis_title":"省份总业绩额（万元）",
        "legend":dict(title=dict(text="象限"),orientation="v",x=1.02,y=0.98,
                      bgcolor="rgba(255,255,255,.92)",bordercolor="#E0E0E0",borderwidth=1,itemsizing="constant"),
    }))
    st.plotly_chart(fig_pot,use_container_width=True)
    st.caption("气泡大小代表当年该省城投债发行数量（只），气泡越大代表融资动能越强。")

    # ── 全司财力与业绩分析的象限概况表 ──
    _qs = scatter_df.groupby("象限").agg(
        各象限总业绩=("业绩总额","sum"),
        各象限成单总数=("成单总数","sum"),
    ).reset_index()
    _qs["各象限总业绩占比"] = (_qs["各象限总业绩"] / _qs["各象限总业绩"].sum() * 100).round(1)
    _qs["各象限每单均额"] = (_qs["各象限总业绩"] / _qs["各象限成单总数"]).round(1)
    _qp = df_stats.merge(scatter_df[["省份","象限"]], on="省份", how="inner")
    _qp_cnt = _qp.groupby("象限")["市场人员"].nunique().reset_index()
    _qp_cnt.columns = ["象限","各象限展业人数"]
    _qs = _qs.merge(_qp_cnt, on="象限", how="left")
    _qs["各象限人均业绩"] = (_qs["各象限总业绩"] / _qs["各象限展业人数"]).round(1)
    _qs = _qs.sort_values("各象限总业绩", ascending=False)
    _qs["各象限总业绩占比"] = _qs["各象限总业绩占比"].apply(lambda v: f"{v:.1f}%")
    st.markdown("**全司四象限概况**")
    st.dataframe(
        _qs[["象限","各象限总业绩","各象限总业绩占比","各象限成单总数","各象限每单均额","各象限展业人数","各象限人均业绩"]],
        use_container_width=True, hide_index=True,
        column_config={
            "象限": st.column_config.TextColumn("象限", width="small"),
            "各象限总业绩": st.column_config.NumberColumn("各象限总业绩(万)", format="%.1f"),
            "各象限总业绩占比": st.column_config.TextColumn("各象限总业绩占比", width="small"),
            "各象限成单总数": st.column_config.NumberColumn("各象限成单总数(笔)", format="%d"),
            "各象限每单均额": st.column_config.NumberColumn("各象限每单均额(万)", format="%.1f"),
            "各象限展业人数": st.column_config.NumberColumn("各象限展业人数", format="%d"),
            "各象限人均业绩": st.column_config.NumberColumn("各象限人均业绩(万)", format="%.1f"),
        })

    # ── 省域成交模式分析·攻打战术
    sec("省域成交模式分析·攻打战术")
    
    st.markdown(
        "<div style='font-size:.84rem;color:#555;line-height:1.8;margin-bottom:10px'>"   
        "本模块通过成单数（市场渗透率）× 单笔均额（利润空间）象限分析，将各省份归入四种成交模式类型，指导具体打法设计。<br>"
        "</div>",unsafe_allow_html=True)
    # 紧凑列宽：method选择框只占1/4
    mc1,mc2,mc3,mc4=st.columns([1,3,3,3])
    with mc1:
        st.markdown("<span style='font-size:.82rem;color:#555;font-weight:500'>请选择分析方法：</span>",unsafe_allow_html=True)
        method=st.selectbox("分析方法",["聚类分析","参数设定"],key="market_method",label_visibility="collapsed")

    METHOD_DESC={
        "聚类分析":"聚类分析法原理：利用数学距离计算样本间的差异，进行特征空间划分，自动挖掘数据内部隐藏的逻辑结构与自然分布。核心逻辑是「物以类聚」。聚类分析中存在最优聚类数量 k，但数学最优解并不总等同于业务最优解，因此开放 k 值设置，允许在逻辑底座上根据长期观察与业务经验进行微调。",
        "参数设定":"参数设定法原理：基于单笔均额与成单数的双维度象限划分。横轴（成单数）衡量市场渗透率；纵轴（单笔均额）衡量利润空间与专业壁垒。",
    }
    st.markdown(f"<div style='font-size:.84rem;color:#555;margin-bottom:10px'>{METHOD_DESC[method]}</div>",unsafe_allow_html=True)

    t_cols=st.columns(4)
    for (name, color, sub, desc, strat), col in zip(TYPE_4_DESCS, t_cols):
        col.markdown(
            f"<div class='type-card' style='border-left: 4px solid {color}; padding-left: 10px;'>"
            f"<h4 style='color:{color}; margin-bottom: 5px;'>{name}</h4>"
            f"<p style='color:#777; font-size:.72rem; margin-bottom:3px;'>{sub}</p>"
            f"<p style='font-size:.76rem; margin-bottom:8px;'>{desc}</p>"
            # --- 下面这一行是新增的，用来显示 strat ---
            f"<p style='font-size:.72rem; color:{color}; border-top:1px dotted #eee; pt:5px;'><b>建议：</b>{strat}</p>"
            f"</div>",
            unsafe_allow_html=True
        )
    if method=="聚类分析":
        #最小值 2：聚类至少要有两个组（对比才有意义）。
        #最大值 6：这是一个典型的管理幅度限制。在区域经济或市场分析中，如果分出 7 个以上的类别，人类大脑很难快速建立认知，分类会变得过于细碎，失去"战略指导"意义。
        #默认值 6：最大分类颗粒度，提供最丰富的市场细分视角，用户可根据需要向下调整。
        n_clus=st.slider("聚类数量 k（可选择）",2,6,6,key="k_clus")
        st.caption("K-Means聚类产生k个簇，每个簇按中心坐标归入n种市场类型。多个簇可能归入同一类型，因此实际显示的类型数n≤簇数k。")
        df_clus=cluster_provinces(df_stats,n_clus)
        df_clus["显示类型"]=df_clus["市场类型"]
        x_mid=df_clus["成单数"].median(); y_mid=df_clus["单笔均额(万元)"].median()
        plot_df=df_clus; type_col="显示类型"
    else:
        p50p=float(df_stats["单笔均额(万元)"].median())
        p50c=float(df_stats["成单数"].median())
        cs1,cs2=st.columns(2)
        with cs1:
            thr_price=st.slider("单笔均额（万元）",0.,float(df_stats["单笔均额(万元)"].max()),round(p50p,1),step=1.,key="thr_p")
            st.caption(f"全司所有'人员-省份'组合的单笔均额的中位数 = {p50p:.1f}万")
        with cs2:
            thr_count=st.slider("成单数（笔）",1,int(df_stats["成单数"].max()),max(1,int(p50c)),step=1,key="thr_c")
            st.caption(f"全司所有'人员-省份'组合的成单数的中位数 = {p50c:.0f}笔")
        prov_p=cluster_provinces(df_stats,4).copy()
        def plabel(r):
            hp=r["单笔均额(万元)"]>=thr_price; hc=r["成单数"]>=thr_count
            return "大客户维护型" if hp and not hc else("散户扩张型" if not hp and hc else("优质核心市场" if hp and hc else "边缘待评估市场"))
        prov_p["显示类型"]=prov_p.apply(plabel,axis=1)
        x_mid=thr_count; y_mid=thr_price; plot_df=prov_p; type_col="显示类型"

    ALL_4_TYPES=["大客户维护型","优质核心市场","边缘待评估市场","散户扩张型"]
    x_max_m=plot_df["成单数"].max()*1.3; y_max_m=plot_df["单笔均额(万元)"].max()*1.35
    ms_m=max(plot_df["业绩总额"].max(),1)

    fig_mkt=go.Figure()
    Q_BG_M=[
        ("rgba(21,101,192,.06)",  0,x_mid,   y_mid,y_max_m),
        ("rgba(106,27,154,.06)",  x_mid,x_max_m,y_mid,y_max_m),
        ("rgba(109,76,65,.06)",   0,x_mid,   0,y_mid),
        ("rgba(46,125,50,.06)",   x_mid,x_max_m,0,y_mid),
    ]
    Q_LBL_M=[("大客户维护型",x_mid*.35,(y_mid+y_max_m)/2),("优质核心市场",(x_mid+x_max_m)/2,(y_mid+y_max_m)/2),
             ("边缘待评估市场",x_mid*.35,y_mid*.35),("散户扩张型",(x_mid+x_max_m)/2,y_mid*.35)]
    for fill,x0,x1,y0,y1 in Q_BG_M:
        fig_mkt.add_shape(type="rect",x0=x0,x1=x1,y0=y0,y1=y1,fillcolor=fill,line_width=0,layer="below")
    for lbl,lx,ly in Q_LBL_M:
        fig_mkt.add_annotation(x=lx,y=ly,text=lbl,showarrow=False,font=dict(size=10,color="rgba(0,0,0,.18)"),xanchor="center")
    fig_mkt.add_hline(y=y_mid,line_dash="dash",line_color="#90A4AE",line_width=1,
                      annotation_text=f"单笔均额中位 {y_mid:.1f}万",annotation_position="right")
    fig_mkt.add_vline(x=x_mid,line_dash="dash",line_color="#90A4AE",line_width=1,
                      annotation_text=f"成单数中位 {x_mid}笔",annotation_position="top")

    for t in ALL_4_TYPES:
        sub=plot_df[plot_df[type_col]==t]
        if sub.empty: continue
        color=MKT_COLORS.get(t,"#607D8B")
        sizes=sub["业绩总额"]/ms_m*30+10
        td=TYPE_DESC_MAP.get(t,{})
        ht_lines=[f"<b>%{{text}}【{t}】</b>",
                  f"业绩总额：%{{customdata[0]:.1f}}万元",
                  f"成单数量：%{{x:.0f}}单",
                  f"每单均额：%{{y:.1f}}万元",
                  f"---",
                  f"省域成交模式分析·攻打战术：{td.get('sub','')}",
                  f"{td.get('desc','')}",
                  f"{td.get('strat','')}",
                  f"<extra></extra>"]
        fig_mkt.add_trace(go.Scatter(
            x=sub["成单数"],y=sub["单笔均额(万元)"],
            mode="markers+text",text=sub["省份"],
            textposition="top center",textfont=dict(size=9),
            marker=dict(size=sizes,color=color,opacity=.87,line=dict(color="white",width=1)),
            name=t,
            customdata=sub[["业绩总额"]].values,
            hovertemplate="<br>".join(ht_lines),
        ))

    fig_mkt.update_layout(chart_layout(480, 0.98, {
        "xaxis_title": "成单数", "yaxis_title": "每单业绩均额（万元）",
        "legend": dict(title=dict(text="市场类型"), orientation="v", x=1.02, y=0.98,
                    bgcolor="rgba(255,255,255,.92)", bordercolor="#E0E0E0", borderwidth=1, itemsizing="constant"),
    }))
    st.plotly_chart(fig_mkt,use_container_width=True)

    # ── 市场人员作战画像分析
    sec("市场人员作战画像分析")
    st.markdown(
        "<div style='font-size:.84rem;color:#555;line-height:1.8;margin-bottom:12px'>"
        "本模块采用「穿透式归因」：系统自动识别每笔业绩发生的省份，将其回溯至坐标系中。"
        "通过分析市场人员「拿到结果」的战场类型，判定每个市场人员的人才类型。以下是4大主要类型画像。另5类画像参考选择人员后的<综合战场偏好指数>。"
        "</div>",unsafe_allow_html=True)

    PERSON_ENV_DESCS=[
        ("开疆拓土型",C_QB,"落在【深耕奇迹区】左上🔵","在逆风局中拿到极具含金量的战果，具备极强的个人专业壁垒，认定为「高Alpha人才」，是公司进入新市场的首选人才。"),
        ("核心引擎型",C_QG,"落在【成熟粮仓区】右上🟢","处于公司业务核心腹地，在资源最肥沃的地带执行任务。重点监控费效比，防止高额商务成本换取本属平台的红利。"),
        ("低位蛰伏型",C_QBROWN,"落在【边缘维持区】左下🟤","新人孵化期属于合理的「新手村」；老员工则在处理公司长尾市场。只要费用控制极低即为成功的低成本防御。"),
        ("资源错配型",C_QRED,"落在【高潜低收区】右下🔴","有子弹、没战果。若气泡大而业绩低，可能无法应对大客户或在准入公关中掉队。判定为「资源浪费型」。"),
    ]
    pe_cols=st.columns(4)
    for (name,color,pos,desc),col in zip(PERSON_ENV_DESCS,pe_cols):
        col.markdown(
            f"<div class='type-card' style='border-left-color:{color}'>"
            f"<h4 style='color:{color}'>{name}</h4>"
            f"<p style='color:#777;font-size:.72rem;margin-bottom:3px'>{pos}</p>"
            f"<p style='font-size:.76rem'>{desc}</p></div>",unsafe_allow_html=True)

    # 人员选择：按当年业绩降序，格式：人员（业绩XX万元）
    person_revenue_yr=(df_year.groupby("市场人员")["业绩(万元)"].sum()
                       .reset_index().sort_values("业绩(万元)",ascending=False))
    person_opts_list=["全部（所有人员）"]+[f"{r['市场人员']}（业绩{r['业绩(万元)']:.1f}万元）" for _,r in person_revenue_yr.iterrows()]
    person_raw_list=["全部（所有人员）"]+person_revenue_yr["市场人员"].tolist()
    col_pe_f,_=st.columns([1,3])
    with col_pe_f:
        pe_idx=st.selectbox("按人员过滤业务分布（请下拉选择或手动输入）",range(len(person_opts_list)),
                            format_func=lambda i:person_opts_list[i],key="pe_filter", index=1)
    person_filter=person_raw_list[pe_idx]

    attr_df=scatter_df.copy()
    if person_filter!="全部（所有人员）":
        person_provs=set(df_year[df_year["市场人员"]==person_filter]["省份"].dropna().unique())
        attr_df["active"]=attr_df["省份"].isin(person_provs)
        p_det=(df_year[df_year["市场人员"]==person_filter]
               .groupby("省份").agg(个人业绩=("业绩(万元)","sum"),个人单数=("业绩(万元)","count")).reset_index())
        p_det["个人均额"]=(p_det["个人业绩"]/p_det["个人单数"]).round(1)
        attr_df=attr_df.merge(p_det,on="省份",how="left")
    else:
        attr_df["active"]=True
        attr_df["个人业绩"]=np.nan; attr_df["个人单数"]=np.nan; attr_df["个人均额"]=np.nan

    def attr_hover(r):
        base=(f"<b>{r['省份']}【{r['象限']}】</b><br>"
              f"地方财力：{r['财力(亿元)']:.0f}亿元（全国排名第{r['财力排名']:.0f}）<br>"
              f"当年发行：{int(r['债券发行数量(只)'])}只（全国排名第{r['发行排名']:.0f}）<br>"
              f"全司业绩：{r['业绩总额']:.1f}万元")
        if person_filter!="全部（所有人员）" and pd.notna(r.get("个人业绩")):
            base+=f"<br>---<br>{person_filter} 在此：{r['个人业绩']:.1f}万 / {int(r['个人单数'])}单 / 均{r['个人均额']}万"
        return base
    attr_df["attr_hover"]=attr_df.apply(attr_hover,axis=1)

    # ── 提前计算：当筛选了人员时，先算好重心坐标和画像，供图表使用
    pre_person_total = 0.0
    pre_cx = pre_cy = 0.0
    pre_active_with_perf = None

    if person_filter != "全部（所有人员）":
        _p_active_pre = attr_df[attr_df["active"] & attr_df["个人业绩"].notna() & (attr_df["个人业绩"] > 0)].dropna(subset=["财力(亿元)"])
        if not _p_active_pre.empty:
            _pre_df = _p_active_pre.copy()
            _pre_df["x_coord"] = _pre_df["省份"].map(fiscal_pct_map) * 2 - 1
            _w = _pre_df["个人业绩"]
            _W_pre = _w.sum()
            pre_cx = np.clip((_pre_df["x_coord"] * _w).sum() / _W_pre * CENTROID_SCALE, -1, 1)
            # Y轴：个人总业绩在全公司人员中的绝对百分位（全国展业，市场选择=个人能力）
            _person_total_pct = (all_person_totals < _W_pre).mean()
            pre_cy = np.clip(_person_total_pct * 2 - 1, -1, 1) * CENTROID_SCALE
            pre_person_total = _W_pre
        pre_active_with_perf = _p_active_pre.copy()

    fig_attr=go.Figure()
    for fill,x0,x1,y0,y1 in POT_BG:
        fig_attr.add_shape(type="rect",x0=x0,x1=x1,y0=y0,y1=y1,fillcolor=fill,line_width=0,layer="below")
    for lbl,lx,ly in POT_ANN:
        fig_attr.add_annotation(x=lx,y=ly,text=lbl,showarrow=False,font=dict(size=11,color="rgba(0,0,0,.18)"),xanchor="center")
    fig_attr.add_hline(y=perf_avg_s,line_dash="dash",line_color="#90A4AE",line_width=1,
                       annotation_text=f"省业绩中位 {perf_avg_s:.0f}万",annotation_position="right")
    fig_attr.add_vline(x=fis_mid_s,line_dash="dash",line_color="#90A4AE",line_width=1,
                       annotation_text=f"财力中位 {fis_mid_s:.0f}亿",annotation_position="top")
    for quad,color in POTENCY_COLORS.items():
        fig_attr.add_trace(go.Scatter(x=[None],y=[None],mode="markers",
            marker=dict(size=10,color=color),name=quad,showlegend=True))
    inactive=attr_df[~attr_df["active"]].dropna(subset=["财力(亿元)"])
    if not inactive.empty:
        fig_attr.add_trace(go.Scatter(x=inactive["财力(亿元)"],y=inactive["业绩总额"],mode="markers",
            marker=dict(size=inactive["气泡大小"],color="#CCCCCC",opacity=.25,line=dict(color="white",width=1)),
            name="其他省份",showlegend=True,hovertext=inactive["attr_hover"],hoverinfo="text"))
    # ── 菱形气泡：个人在各省的业绩，位于圆形气泡下方，颜色与象限一致
    if person_filter != "全部（所有人员）" and pre_active_with_perf is not None and not pre_active_with_perf.empty:
        max_personal = pre_active_with_perf["个人业绩"].max()
        diamond_offset = y_max_s * 0.05
        for quad in ["深耕奇迹区","成熟粮仓区","边缘维持区","高潜低收区"]:
            sub_d = pre_active_with_perf[pre_active_with_perf["象限"] == quad]
            if sub_d.empty: continue
            diamond_sizes = sub_d["个人业绩"] / max_personal * 26 + 8
            diamond_hover = sub_d.apply(lambda r: (
                f"<b>{person_filter} · {r['省份']}</b><br>"
                f"个人业绩：{r['个人业绩']:.1f}万元<br>"
                f"成单：{int(r['个人单数'])}单  均{r['个人均额']:.1f}万<br>"
                f"所属象限：{r['象限']}"), axis=1)
            fig_attr.add_trace(go.Scatter(
                x=sub_d["财力(亿元)"],
                y=sub_d["业绩总额"] - diamond_offset,
                mode="markers",
                marker=dict(symbol="diamond",size=diamond_sizes,
                            color=POTENCY_COLORS[quad],opacity=0.87,
                            line=dict(color="white",width=1.5)),
                name=f"{person_filter}个人业绩",
                showlegend=False,
                hovertext=diamond_hover,
                hoverinfo="text"))


    if person_filter != "全部（所有省份）":
        active_df2=attr_df[attr_df["active"]].dropna(subset=["财力(亿元)"])
        for quad in ["深耕奇迹区","成熟粮仓区","边缘维持区","高潜低收区"]:
            sub=active_df2[active_df2["象限"]==quad]
            if sub.empty: continue
            fig_attr.add_trace(go.Scatter(x=sub["财力(亿元)"],y=sub["业绩总额"],mode="markers+text",
                text=sub["省份"],textposition="top center",textfont=dict(size=9),
                marker=dict(size=sub["气泡大小"],color="#B0B0B0",opacity=.55,line=dict(color="white",width=1.5)),
                name=quad,showlegend=False,hovertext=sub["attr_hover"],hoverinfo="text"))


    # ── 综合战场偏好指数的重心坐标映射到图表坐标系
    if person_filter != "全部（所有人员）" and pre_person_total > 0:
        x_half = x_max_s * 0.25
        y_half = y_max_s * 0.25
        centroid_x = fis_mid_s + pre_cx * x_half
        centroid_y = perf_avg_s + pre_cy * y_half

        _p = get_persona(pre_cx, pre_cy)
        centroid_hover = (
            f"<b>{person_filter}（总业绩{pre_person_total:.1f}万元）</b><br>"
            f"综合战场偏好指数<br>---<br>"
            f"{_p['icon']} <b>{_p['label']}</b><br>"
            f"<b>画像定性</b>：{_p['definition']}<br><br>"
            f"<b>管理动作</b>：<br>"
            + "<br>".join(f"• {m}" for m in _p["management"])
            + f"<br><br><i>重心坐标 (X={pre_cx:.2f}, Y={pre_cy:.2f})</i><br>"
            f"<i>X：正=高财力省集中，负=低财力省集中（连续百分位）</i><br>"
            f"<i>Y：正=个人业绩高，负=个人业绩低（全司百分位）</i>"
        )
        fig_attr.add_trace(go.Scatter(
            x=[centroid_x], y=[centroid_y],
            mode="markers+text",
            marker=dict(symbol="triangle-up",size=18,color="#FF6F00",opacity=1.0,
                        line=dict(color="white",width=2)),
            text=[f"{person_filter}"],
            textposition="bottom center",
            textfont=dict(size=9,color="#FF6F00"),
            name=f"重心 ({pre_cx:.2f},{pre_cy:.2f})",
            showlegend=True,
            hovertext=centroid_hover,
            hoverinfo="text"))

    if person_filter == "全部（所有人员）":
        # 全部人员：重建图表，X=加权财力(亿元) Y=个人业绩(万元)，只显示人员
        _all_persons = df_stats["市场人员"].unique()
        _all_data = []  # (px, py, name, color, label, hover)
        for _person in _all_persons:
            _pp = df_stats[df_stats["市场人员"] == _person].merge(
                scatter_df[["省份","财力(亿元)"]], on="省份", how="inner"
            ).dropna(subset=["财力(亿元)"])
            if _pp.empty or _pp["业绩总额"].sum() == 0:
                continue
            _w = _pp["业绩总额"]
            _wt = _w.sum()
            _px = (_pp["财力(亿元)"] * _w).sum() / _wt
            _py = _wt
            _pp["_xc"] = _pp["省份"].map(fiscal_pct_map) * 2 - 1
            _pcx = np.clip((_pp["_xc"] * _w).sum() / _wt * CENTROID_SCALE, -1, 1)
            # Y轴：个人总业绩在全公司人员中的绝对百分位
            _person_total_pct = (all_person_totals < _wt).mean()
            _pcy = np.clip(_person_total_pct * 2 - 1, -1, 1) * CENTROID_SCALE
            _ppersona = get_persona(_pcx, _pcy)
            _all_data.append((
                _px, _py, _person, _ppersona["color"], _ppersona["label"],
                f"<b>{_person}</b><br>"
                f"{_ppersona['icon']} {_ppersona['label']}<br>"
                f"画像定性：{_ppersona['definition']}<br>"
                f"总业绩：{_wt:.1f}万元<br>"
                f"加权平均财力：{_px:.1f}亿元<br>"
                f"重心坐标 (X={_pcx:.2f}, Y={_pcy:.2f})"))
        fig_attr = go.Figure()
        # 按画像类型分组，按价值从高到低排序加入图例
        _LEGEND_ORDER = [
            "核心引擎型","开疆拓土型","全域高效型",
            "顺势收益型","逆境生存型","平衡稳健型",
            "低位蛰伏型","全域待提升型","资源错配型",
        ]
        _grouped = {}
        for _d in _all_data:
            _label = _d[4]
            _grouped.setdefault(_label, []).append(_d)
        _px_all, _py_all = [], []
        for _label in _LEGEND_ORDER:
            if _label not in _grouped:
                continue
            _items = _grouped[_label]
            _xs = [d[0] for d in _items]
            _ys = [d[1] for d in _items]
            _ns = [d[2] for d in _items]
            _hs = [d[5] for d in _items]
            _px_all.extend(_xs)
            _py_all.extend(_ys)
            fig_attr.add_trace(go.Scatter(
                x=_xs, y=_ys,
                mode="markers+text",
                marker=dict(size=14, color=_items[0][3], opacity=0.82,
                            line=dict(color="white", width=1.5)),
                text=_ns,
                textposition="top center",
                textfont=dict(size=9),
                name=_label, showlegend=True,
                legendgroup=_label,
                hovertext=_hs, hoverinfo="text"))
        _py_median = float(np.median(_py_all)) if _py_all else 0
        _px_max = max(_px_all) * 1.15 if _px_all else 10000
        _py_max = max(_py_all) * 1.5 if _py_all else 1000
        fig_attr.add_vline(x=fis_mid_s, line_dash="dash", line_color="#90A4AE", line_width=1,
                           annotation_text=f"财力中位 {fis_mid_s:.0f}亿", annotation_position="top")
        fig_attr.add_hline(y=_py_median, line_dash="dash", line_color="#90A4AE", line_width=1,
                           annotation_text=f"个人业绩中位 {_py_median:.0f}万", annotation_position="right")
        fig_attr.update_layout(
            xaxis_title="加权平均地方财力（亿元）", yaxis_title="个人业绩总额（万元）",
            xaxis=dict(range=[0, _px_max]), yaxis=dict(range=[0, _py_max]),
            height=500, plot_bgcolor="#FDFCFB", paper_bgcolor="#FDFCFB",
            font=dict(family=FONT, size=12),
            margin=dict(l=10, r=160, t=40, b=10),
            legend=dict(title=dict(text="人员类型"), orientation="v", x=1.02, y=0.98,
                        bgcolor="rgba(255,255,255,.92)", bordercolor="#E0E0E0", borderwidth=1,
                        itemsizing="constant"),
            hoverlabel=dict(align="left"))
        st.plotly_chart(fig_attr, use_container_width=True)
    else:
        fig_attr.update_layout(chart_layout(550, 0.98, {
            "xaxis_title":"省份一般公共预算收入（亿元）","yaxis_title":"省份总业绩额（万元）",
            "legend":dict(title=dict(text="象限"),orientation="v",x=1.02,y=0.98,
                          bgcolor="rgba(255,255,255,.92)",bordercolor="#E0E0E0",borderwidth=1,itemsizing="constant"),
        }))
        st.plotly_chart(fig_attr, use_container_width=True)

    # 图表下方说明文字
    if person_filter != "全部（所有人员）":
        st.markdown(
            f"<div style='font-size:.78rem;color:#666;background:#F9F8F6;border-radius:6px;"
            f"padding:8px 14px;margin-top:-8px;line-height:1.9'>"
            f"<b>圆形气泡</b>：圆形气泡大小由该省当年债券发行数量决定。&emsp;"
            f"<b>菱形气泡</b>：<span style='font-weight:600'>{person_filter}</span> 个人在该省份的业绩总额，"
            f"气泡越大=个人业绩越多，颜色与所属象限一致，紧贴圆形气泡下方。&emsp;"
            f"<b>橙色三角形 ▲</b>：综合战场偏好指数的加权重心，"
            f"代表该人员整体展业模式在财力-效率坐标系中的位置，悬停查看完整画像分析。"
            f"</div>",
            unsafe_allow_html=True)
    st.markdown("")
    # 战场分布表 + 综合战场偏好指数
    if person_filter!="全部（所有人员）":
        p_active=attr_df[attr_df["active"]&(attr_df["业绩总额"]>0)].dropna(subset=["财力(亿元)"])
        if not p_active.empty:
            # 个人在各象限的业绩
            p_quad_grp=p_active.groupby("象限").apply(lambda d: pd.Series({
                "省份数":len(d),
                "个人业绩":d["个人业绩"].sum() if "个人业绩" in d.columns else 0,
            })).reset_index()
            # 全司各象限业绩：从全量省份计算，不随人员筛选变化
            _quad_total = scatter_df.groupby("象限")["业绩总额"].sum().reset_index()
            _quad_total.columns = ["象限","全司各象限业绩"]
            p_quad_grp = p_quad_grp.merge(_quad_total, on="象限", how="left")
            p_quad_grp["全司各象限业绩"] = p_quad_grp["全司各象限业绩"].fillna(0)
            p_quad_grp["个人贡献度"]=p_quad_grp.apply(
                lambda r: r["个人业绩"]/r["全司各象限业绩"]*100 if r["全司各象限业绩"]>0 else 0, axis=1)
            avg_contrib=p_quad_grp["个人贡献度"].mean()
            # 个人象限排名：该人员在各象限的业绩在同象限所有人员中的排名
            _all_quad = df_stats.merge(
                scatter_df[["省份","象限"]], on="省份", how="inner"
            ).groupby(["象限","市场人员"])["业绩总额"].sum().reset_index()
            _all_quad["象限排名"] = _all_quad.groupby("象限")["业绩总额"].rank(
                ascending=False, method="min"
            ).astype(int)
            _all_quad["象限总人数"] = _all_quad.groupby("象限")["市场人员"].transform("nunique")
            _pr = _all_quad[_all_quad["市场人员"] == person_filter][["象限","象限排名","象限总人数"]]
            p_quad_grp = p_quad_grp.merge(_pr, on="象限", how="left")
            p_quad_grp["个人象限排名"] = p_quad_grp["象限排名"].fillna(-1).astype(int)
            p_quad_grp["象限总人数"] = p_quad_grp["象限总人数"].fillna(0).astype(int)

            # ── 综合战场偏好指数（加权重心模型，连续坐标）
            _cd = p_active.copy()
            _cd["x_coord"] = _cd["省份"].map(fiscal_pct_map) * 2 - 1
            _w2 = _cd["个人业绩"]
            _W2 = _w2.sum()
            if _W2 > 0:
                cx = np.clip((_cd["x_coord"] * _w2).sum() / _W2 * CENTROID_SCALE, -1, 1)
                # Y轴：个人总业绩在全公司人员中的绝对百分位
                _person_total_pct = (all_person_totals < _W2).mean()
                cy = np.clip(_person_total_pct * 2 - 1, -1, 1) * CENTROID_SCALE
            else:
                cx = cy = 0.0

            persona = get_persona(cx, cy)
            pref_label=f"{persona['icon']} {persona['label']}"
            pref_color=persona["color"]
            pref_desc=(
                       f"<b>画像定性</b>：{persona['definition']}<br><br>"
                       f"<b>管理动作</b>：<br>"
                       +"<br>".join(f"• {m}" for m in persona["management"])
                       +f"<br><br><i style='color:#888;font-size:.78rem'>"
                       f"重心坐标 (X={cx:.2f}, Y={cy:.2f}) | "
                       f"X轴：财力集中度（正=高财力，负=低财力）| "
                       f"Y轴：个人业绩水平（正=高产，负=低产，全司百分位）</i>")

            # 左：表格  右：偏好指数
            tbl_col,idx_col=st.columns([1,1])
            with tbl_col:
                st.markdown(f"**{person_filter} 的战场分布**")
                _bf = p_quad_grp[["象限","个人业绩","全司各象限业绩","个人贡献度","个人象限排名","象限总人数"]].copy()
                _bf = _bf.sort_values("个人业绩", ascending=False)
                _bf["个人各象限业绩(万)"] = _bf["个人业绩"].apply(lambda v: f"{v:.1f}" if pd.notna(v) else "-")
                _bf["全司各象限业绩(万)"] = _bf["全司各象限业绩"].round(1)
                _bf["个人对象限贡献度"] = _bf.apply(
                    lambda r: f"{r['个人贡献度']:.1f}%{' ⭐' if r['个人贡献度'] > avg_contrib else ''}", axis=1)
                _bf["个人象限排名"] = _bf.apply(
                    lambda r: f"第{int(r['个人象限排名'])}/共{int(r['象限总人数'])}"
                    if r['个人象限排名'] > 0 else "-", axis=1)
                st.dataframe(
                    _bf[["象限","个人各象限业绩(万)","全司各象限业绩(万)","个人对象限贡献度","个人象限排名"]],
                    use_container_width=True, hide_index=True,
                    column_config={
                        "象限": st.column_config.TextColumn("象限", width="small"),
                        "个人各象限业绩(万)": st.column_config.TextColumn("个人各象限业绩(万)", width="small"),
                        "全司各象限业绩(万)": st.column_config.NumberColumn("全司各象限业绩(万)", format="%.1f"),
                        "个人对象限贡献度": st.column_config.TextColumn("个人对象限贡献度(%)", width="small"),
                        "个人象限排名": st.column_config.TextColumn("个人象限排名", width="small"),
                    })
                st.caption("⭐=贡献度高于均值（核心长板）| 排名=同象限展业人员中的位次（第X/共Y）| 点击列头排序")

            with idx_col:
                st.markdown(f"**{person_filter} 的综合战场偏好指数**")
                st.markdown(
                    f"<div style='border-left:4px solid {pref_color};background:{pref_color}10;border-radius:8px;"
                    f"padding:14px 16px;margin-top:4px'>"
                    f"<div style='font-size:1.05rem;font-weight:700;color:{pref_color};margin-bottom:10px'>{pref_label}</div>"
                    f"<div style='font-size:.82rem;color:#444;line-height:1.9'>{pref_desc}</div>"
                    f"</div>",unsafe_allow_html=True)

    st.markdown('<div class="caption-box">人员的逻辑重心x轴使用财力百分位，y轴使用个人业绩水平百分位，权重为各省份业绩。</div>',unsafe_allow_html=True)


# ═══════════════════════════════════════════════════════
# 模块二：部门人效分析
# ═══════════════════════════════════════════════════════
else:
    st.markdown(f"<h2 style='color:{C_TITLE};margin-bottom:4px'>部门人效分析</h2>",unsafe_allow_html=True)
    st.caption("业务一部 · 2023年1-10月 · 费用来源：报销明细 + 携程商旅（不含工资奖金社保）")

    rev_bp=df_rev.groupby("市场人员")["金额"].sum().reset_index(); rev_bp.columns=["市场人员","收入合计(元)"]
    exp_bp=df_exp.groupby("市场人员")["金额"].sum().reset_index(); exp_bp.columns=["市场人员","费用合计(元)"]
    df_dept=pd.merge(rev_bp,exp_bp,on="市场人员",how="outer").fillna(0)
    df_dept["费用/收入"]=df_dept.apply(lambda r:r["费用合计(元)"]/r["收入合计(元)"] if r["收入合计(元)"]>0 else np.nan,axis=1)
    df_dept["费用上限(2%)"]=df_dept["收入合计(元)"]*0.02
    df_dept["超支额(元)"]=df_dept["费用上限(2%)"]-df_dept["费用合计(元)"]

    total_rev=df_dept["收入合计(元)"].sum(); total_exp=df_dept["费用合计(元)"].sum()
    dept_ratio=total_exp/total_rev if total_rev>0 else 0
    efficiency=total_rev/total_exp if total_exp>0 else 0

    arr22=df_arr[df_arr["年度"]=="2022年"]
    per_p22=arr22.groupby("市场人员")["业绩(万元)"].sum()
    company_monthly_avg=float((per_p22/12).mean())
    company_order_avg=float(arr22["业绩(万元)"].mean())
    dept_order_avg=total_rev/len(df_rev)/10000
    zero_rev_persons=df_dept[df_dept["收入合计(元)"]==0]["市场人员"].tolist()

    # ── 部门效能总览
    sec("部门效能总览")
    c1,c2,c3,c4,c5=st.columns(5)
    c1.metric("部门总业绩",f"{total_rev/10000:.1f} 万元")
    c2.metric("部门总费用",f"{total_exp/10000:.1f} 万元")
    c3.metric("获客成本率",f"{dept_ratio:.2%}",
              delta=f"{'超出' if dept_ratio>0.02 else '达标'}目标2%",
              delta_color="inverse" if dept_ratio>0.02 else "normal")
    c4.metric("获客效能比",f"{efficiency:.1f}x",
              delta=f"公司均值50x，{'低于' if efficiency<50 else '高于'}均值",delta_color="normal")
    c5.metric("部门平均单笔业绩",f"{dept_order_avg:.1f} 万元",
              delta=f"全司均值 {company_order_avg:.1f}万",delta_color="normal")
    st.markdown(
        f"<div style='font-size:.8rem;color:#555;background:#F5F2F0;border-radius:8px;padding:10px 16px;margin-top:6px;line-height:2'>"
        f"<b>获客效能比</b>：每投入1元展业费用所带来的收入。按2%获客成本率，公司平均获客效能比为 <b style='color:{C_TITLE}'>50x</b>，部门当前为 <b style='color:{C_TITLE}'>{efficiency:.1f}x</b>，{'<b style=\"color:#C62828\">低于</b>' if efficiency<50 else '<b style=\"color:#2E7D32\">高于</b>'}公司均值。<br>"
        f"<b>部门平均单笔业绩</b>：{dept_order_avg:.1f}万元，{'<b style=\"color:#C62828\">低于</b>' if efficiency<50 else '<b style=\"color:#2E7D32\">高于</b>'}全司2022年平均单笔业绩 <b>{company_order_avg:.1f}</b>万元。"
        f"</div>",unsafe_allow_html=True)

    # ── 部门各人员业绩、费用对比
    sec("部门各人员业绩、费用对比")
    df_sorted=df_dept.sort_values("收入合计(元)",ascending=False)
    persons_order=df_sorted["市场人员"].tolist()
    exp_cat=df_exp.groupby(["市场人员","费用大类"])["金额"].sum().reset_index()
    BW=0.45

    # 两个柱状图并排，下方表格全宽
    ch_l,ch_r=st.columns(2)

    with ch_l:
        st.markdown("**业绩柱状图**")
        rev_vals=df_sorted["收入合计(元)"].values/10000
        fig_rv=go.Figure(go.Bar(
            x=persons_order,y=rev_vals,marker_color=C_TEAL,width=BW,
            text=[f"{int(round(v))}" for v in rev_vals],
            textposition="outside",textfont=dict(size=10),
            hovertemplate="<b>%{x}</b><br>业绩：%{y:.1f}万<extra></extra>"))
        fig_rv.update_layout(xaxis_title="",yaxis_title="业绩额（万元）",height=320,
            plot_bgcolor="#FDFCFB",paper_bgcolor="#FDFCFB",font=dict(family=FONT,size=11),
            margin=dict(l=10,r=10,t=30,b=10),yaxis=dict(range=[0,max(rev_vals)*1.28]))
        st.plotly_chart(fig_rv,use_container_width=True)

    with ch_r:
        st.markdown("**费用柱状图**")
        dl_vals=[exp_cat[(exp_cat["市场人员"]==p)&(exp_cat["费用大类"]=="差旅费")]["金额"].sum()/10000 for p in persons_order]
        zd_vals=[exp_cat[(exp_cat["市场人员"]==p)&(exp_cat["费用大类"]=="招待费")]["金额"].sum()/10000 for p in persons_order]
        tv_vals=[dl+zd for dl,zd in zip(dl_vals,zd_vals)]
        fig_ex=go.Figure()
        # 差旅费、招待费堆叠，用customdata传递总费用和差旅费
        exp_customdata=list(zip(tv_vals,dl_vals,zd_vals))
        fig_ex.add_trace(go.Bar(name="差旅费",x=persons_order,y=dl_vals,marker_color=C_COCOA,width=BW,
            text=None,
            customdata=exp_customdata,
            hovertemplate="<b>%{x}</b><br>费用：%{customdata[0]:.1f}万<br>其中差旅费：%{customdata[1]:.1f}万<br>其中招待费：%{customdata[2]:.1f}万<extra></extra>"))
        fig_ex.add_trace(go.Bar(name="招待费",x=persons_order,y=zd_vals,marker_color=C_PARCH,width=BW,
            text=None,
            customdata=exp_customdata,
            hovertemplate="<b>%{x}</b><br>费用：%{customdata[0]:.1f}万<br>其中差旅费：%{customdata[1]:.1f}万<br>其中招待费：%{customdata[2]:.1f}万<extra></extra>"))
        # 总费用标注在柱顶居中，字体与收入图一致（size=10）
        for i,(p,tv,dl,zd) in enumerate(zip(persons_order,tv_vals,dl_vals,zd_vals)):
            if tv>0.01:
                fig_ex.add_annotation(x=p,y=tv,
                    text=f"{tv:.1f}",showarrow=False,
                    yanchor="bottom",yshift=4,
                    font=dict(size=10,color="#333"),
                    xanchor="center")
        # 费用上限红线
        for i,lv in enumerate(df_sorted["费用上限(2%)"].values/10000):
            fig_ex.add_shape(type="rect",x0=i-BW/2,x1=i+BW/2,y0=lv-.015,y1=lv+.015,
                fillcolor="#C62828",line_color="#C62828",line_width=0,xref="x",yref="y")
        # 图例：费用上限红线（shape不出现在图例中，需加一条空trace占位）
        fig_ex.add_trace(go.Scatter(
            x=[None], y=[None],
            mode="lines",
            line=dict(color="#C62828", width=2),
            name="费用上限(2%)",
            showlegend=True))
        fig_ex.update_layout(barmode="stack",xaxis_title="",yaxis_title="费用（万元）",height=320,
            plot_bgcolor="#FDFCFB",paper_bgcolor="#FDFCFB",font=dict(family=FONT,size=11),
            legend=dict(orientation="h",yanchor="bottom",y=1.02,xanchor="right",x=1),
            margin=dict(l=10,r=10,t=60,b=10))
        st.plotly_chart(fig_ex,use_container_width=True)

    # 表格全宽 — 使用 st.dataframe 支持点击列头排序
    st.markdown("**部门内个人收入及费用表**")
    _dd = df_sorted.copy()
    # 预计算每人差旅费、招待费
    _dl_map = exp_cat[exp_cat["费用大类"]=="差旅费"].groupby("市场人员")["金额"].sum() / 10000
    _zd_map = exp_cat[exp_cat["费用大类"]=="招待费"].groupby("市场人员")["金额"].sum() / 10000
    _dd["差旅费(万)"] = _dd["市场人员"].map(_dl_map).fillna(0).round(2)
    _dd["招待费(万)"] = _dd["市场人员"].map(_zd_map).fillna(0).round(2)
    _dd["业绩(万)"] = (_dd["收入合计(元)"] / 10000).round(2)
    _dd["费用(万)"] = (_dd["费用合计(元)"] / 10000).round(2)
    _dd["费用上限(万)"] = (_dd["费用上限(2%)"] / 10000).round(2)
    _dd["超支/结余(万)"] = (_dd["超支额(元)"] / 10000).round(2)
    _dd["获客成本率"] = _dd["费用/收入"].apply(lambda x: f"{x:.2%}" if pd.notna(x) else "N/A")
    _dd["状态"] = _dd.apply(lambda r:
        "❌ 零收入" if r["收入合计(元)"] == 0
        else ("⚠️ 超支" if r["超支额(元)"] < 0 else "✅ 达标"), axis=1)
    st.dataframe(
        _dd[["市场人员","业绩(万)","费用(万)","差旅费(万)","招待费(万)",
             "费用上限(万)","超支/结余(万)","获客成本率","状态"]],
        use_container_width=True, hide_index=True,
        column_config={
            "市场人员": st.column_config.TextColumn("人员", width="small"),
            "业绩(万)": st.column_config.NumberColumn("业绩（万）", format="%.2f"),
            "费用(万)": st.column_config.NumberColumn("费用（万）", format="%.2f"),
            "差旅费(万)": st.column_config.NumberColumn("差旅费（万）", format="%.2f"),
            "招待费(万)": st.column_config.NumberColumn("招待费（万）", format="%.2f"),
            "费用上限(万)": st.column_config.NumberColumn("费用上限（万）", format="%.2f"),
            "超支/结余(万)": st.column_config.NumberColumn("超支/结余（万）", format="%.2f"),
            "获客成本率": st.column_config.TextColumn("获客成本率", width="small"),
            "状态": st.column_config.TextColumn("状态", width="small"),
        })
    st.caption("获客成本率>2%标⚠️；零收入标❌ | 点击列头升降序")



    # ── 合作关系分析
    sec("合作关系分析")
    all_ps=df_dept["市场人员"].tolist()
    left_col,right_col=st.columns([1,1])

    with left_col:
        sel_r1,sel_r2=st.columns(2)
        with sel_r1: cp1=st.selectbox("人员 A",all_ps,index=0,key="cp1")
        with sel_r2: cp2=st.selectbox("人员 B",[p for p in all_ps if p!=cp1],index=0,key="cp2")

        r1=df_dept[df_dept["市场人员"]==cp1].iloc[0]
        r2=df_dept[df_dept["市场人员"]==cp2].iloc[0]
        c_rev=r1["收入合计(元)"]+r2["收入合计(元)"]
        c_exp=r1["费用合计(元)"]+r2["费用合计(元)"]
        c_ratio=c_exp/c_rev if c_rev>0 else np.nan
        c_limit=c_rev*0.02; c_over=c_limit-c_exp

        # 2x2 指标卡
        mc1,mc2=st.columns(2)
        mc3,mc4=st.columns(2)
        for col,label,val in [(mc1,"合计收入",f"{c_rev/10000:.2f}万元"),
                               (mc2,"合计费用",f"{c_exp/10000:.2f}万元"),
                               (mc3,"合并获客成本率",f"{c_ratio:.2%}" if not np.isnan(c_ratio) else "N/A"),
                               (mc4,"合并超支/结余",f"{c_over/10000:.2f}万元")]:
            col.markdown(
                f"<div class='metric-mini'><div class='lbl'>{label}</div><div class='val'>{val}</div></div>",
                unsafe_allow_html=True)

    with right_col:
        persons_co=[cp1,cp2,"合并"]
        rev_co=[r1["收入合计(元)"]/10000,r2["收入合计(元)"]/10000,c_rev/10000]
        def gec(p,cat):
            try: return df_exp[(df_exp["市场人员"]==p)&(df_exp["费用大类"]==cat)]["金额"].sum()/10000
            except: return 0.
        dl_co=[gec(p,"差旅费") for p in [cp1,cp2]]+[sum(gec(p,"差旅费") for p in [cp1,cp2])]
        zd_co=[gec(p,"招待费") for p in [cp1,cp2]]+[sum(gec(p,"招待费") for p in [cp1,cp2])]
        tv_co=[dl+zd for dl,zd in zip(dl_co,zd_co)]
        lc_co=[r1["费用上限(2%)"]/10000,r2["费用上限(2%)"]/10000,c_limit/10000]
        BW2=0.45

        co_ch_l,co_ch_r=st.columns(2)
        with co_ch_l:
            fig_cr=go.Figure(go.Bar(x=persons_co,y=rev_co,marker_color=C_TEAL,width=BW2,
                text=[f"{int(round(v))}" for v in rev_co],textposition="outside",textfont=dict(size=10),
                hovertemplate="<b>%{x}</b><br>业绩：%{y:.1f}万<extra></extra>"))
            fig_cr.update_layout(xaxis_title="",yaxis_title="业绩额（万元）",height=300,
                plot_bgcolor="#FDFCFB",paper_bgcolor="#FDFCFB",font=dict(family=FONT,size=11),
                margin=dict(l=5,r=5,t=25,b=5),yaxis=dict(range=[0,max(rev_co)*1.3]))
            st.plotly_chart(fig_cr,use_container_width=True)

        with co_ch_r:
            fig_ce=go.Figure()
            co_customdata=list(zip(tv_co,dl_co,zd_co))
            fig_ce.add_trace(go.Bar(name="差旅费",x=persons_co,y=dl_co,marker_color=C_COCOA,width=BW2,
                text=None,
                customdata=co_customdata,
                hovertemplate="<b>%{x}</b><br>费用：%{customdata[0]:.1f}万<br>其中差旅费：%{customdata[1]:.1f}万<br>其中招待费：%{customdata[2]:.1f}万<extra></extra>"))
            fig_ce.add_trace(go.Bar(name="招待费",x=persons_co,y=zd_co,marker_color=C_PARCH,width=BW2,
                text=None,
                customdata=co_customdata,
                hovertemplate="<b>%{x}</b><br>费用：%{customdata[0]:.1f}万<br>其中差旅费：%{customdata[1]:.1f}万<br>其中招待费：%{customdata[2]:.1f}万<extra></extra>"))
            # 总费用标注在柱顶居中，字体与收入图一致（size=10）
            for i,(p,tv) in enumerate(zip(persons_co,tv_co)):
                if tv>0.01:
                    fig_ce.add_annotation(x=p,y=tv,text=f"{tv:.1f}",showarrow=False,
                                          yanchor="bottom",yshift=4,font=dict(size=10,color="#333"),xanchor="center")
            for i,lv in enumerate(lc_co):
                fig_ce.add_shape(type="rect",x0=i-BW2/2,x1=i+BW2/2,y0=lv-.015,y1=lv+.015,
                    fillcolor="#C62828",line_color="#C62828",line_width=0,xref="x",yref="y")
            fig_ce.add_trace(go.Scatter(
                x=[None], y=[None],
                mode="lines",
                line=dict(color="#C62828", width=2),
                name="费用上限(2%)",
                showlegend=True))
            fig_ce.update_layout(barmode="stack",xaxis_title="",yaxis_title="费用（万元）",height=300,
                plot_bgcolor="#FDFCFB",paper_bgcolor="#FDFCFB",font=dict(family=FONT,size=11),
                legend=dict(orientation="h",yanchor="bottom",y=1.02,xanchor="right",x=1),
                margin=dict(l=5,r=5,t=55,b=5))
            st.plotly_chart(fig_ce,use_container_width=True)

    # ── 人员费效象限分析（最后）
    sec("人员费效象限分析")

    PERSON_TYPE_4=[
        ("尖兵型",  C_PB,  "高业绩 · 低费用","资源占用极低但贡献巨大，优先分析其作业链路，提炼可复制的展业方法论，作为团队标准化样板推广。"),
        ("重炮型",C_PG,    "高业绩 · 高费用","高投入带来高产出，可能涉及处理明星项目或核心客户。重点监测费效比，防止高额费用掩盖真实净利润。"),
        ("潜伏型",C_PBROWN,"低业绩 · 低费用","投入产出双低，需区分是新人成长期、市场开拓期，还是能力遭遇瓶颈，给予相应支持或设定观察期后再作判断。"),
        ("失血型",C_PRED,  "低业绩 · 高费用","高资源消耗却未转化业绩，需优先排查原因：是客户周期长导致的阶段性滞后，还是展业方向与自身能力存在错配？若持续处于此区域，应启动管理介入，明确改善目标与时间节点。"),
    ]
    # 构建象限→描述映射，供悬浮框调用
    PTYPE_DESC_MAP={name:{"sub":sub,"desc":desc} for name,_,sub,desc in PERSON_TYPE_4}
    PTYPE_DESC_MAP["待观察"]={"sub":"收入为零","desc":"该人员本统计周期内收入登记为零，可能处于新人培育期或市场开拓期，暂不适用类型评估。"}
    pt_cols=st.columns(4)
    for (name,color,sub,desc),col in zip(PERSON_TYPE_4,pt_cols):
        col.markdown(
            f"<div class='type-card' style='border-left-color:{color}'>"
            f"<h4 style='color:{color}'>{name}</h4>"
            f"<p style='color:#777;font-size:.72rem;margin-bottom:3px'>{sub}</p>"
            f"<p style='font-size:.76rem'>{desc}</p></div>",unsafe_allow_html=True)

    st.caption(
        f"Y轴基准线：2022年全司人均月度业绩均值 ≈ {company_monthly_avg:.1f}万元/月，代表公司「生存线」或「及格线」。\n"
        f"X轴基准线：统计周期内部门人均费用中位数，锚定部门中等资源消耗水平。")

    # 零收入说明（图上方）
    if zero_rev_persons:
        names="、".join(zero_rev_persons)
        st.markdown(
            f'<div class="note-teal"><b>{names}</b> 本统计周期内收入登记为零，但存在费用支出。'
            f'该人员可能处于<b>新人培育期 / 市场开拓期</b>，尚未形成收入，不代表数据错误。'
            f'获客效能比指标暂不适用，在四象限图中以 × 标注。</div>',unsafe_allow_html=True)

    df_quad=df_dept.copy()
    df_quad["月均收入(万元)"]=df_quad["收入合计(元)"]/10/10000
    df_quad["费用合计(万元)"]=df_quad["费用合计(元)"]/10000
    exp_mid_wan=df_quad["费用合计(万元)"].median()

    collab_rev_q=(r1["收入合计(元)"]+r2["收入合计(元)"])/10/10000
    collab_exp_q=(r1["费用合计(元)"]+r2["费用合计(元)"])/10000
    collab_name=f"{cp1}+{cp2}"

    def qlabel(r):
        if r["收入合计(元)"]==0: return "待观察"
        hi_y=r["月均收入(万元)"]>=company_monthly_avg; lo_x=r["费用合计(万元)"]<=exp_mid_wan
        return "尖兵型" if hi_y and lo_x else("重炮型" if hi_y else("潜伏型" if lo_x else "失血型"))
    df_quad["象限"]=df_quad.apply(qlabel,axis=1)
    if collab_rev_q>=company_monthly_avg and collab_exp_q<=exp_mid_wan: cq="尖兵型"
    elif collab_rev_q>=company_monthly_avg: cq="重炮型"
    elif collab_exp_q<=exp_mid_wan: cq="潜伏型"
    else: cq="失血型"

    x_max_q=max(df_quad["费用合计(万元)"].max(),collab_exp_q)*1.4
    y_max_q=max(df_quad["月均收入(万元)"].max(),collab_rev_q)*1.4

    fig_q=go.Figure()
    Q_BG_P=[("rgba(21,101,192,.06)",0,exp_mid_wan,company_monthly_avg,y_max_q),
            ("rgba(46,125,50,.06)",exp_mid_wan,x_max_q,company_monthly_avg,y_max_q),
            ("rgba(109,76,65,.06)",0,exp_mid_wan,0,company_monthly_avg),
            ("rgba(198,40,40,.06)",exp_mid_wan,x_max_q,0,company_monthly_avg)]
    Q_ANN_P=[("尖兵型",exp_mid_wan*.3,(company_monthly_avg+y_max_q)/2),
             ("重炮型",(exp_mid_wan+x_max_q)/2,(company_monthly_avg+y_max_q)/2),
             ("潜伏型",exp_mid_wan*.3,company_monthly_avg*.2),
             ("失血型",(exp_mid_wan+x_max_q)/2,company_monthly_avg*.2)]
    for fill,x0,x1,y0,y1 in Q_BG_P:
        fig_q.add_shape(type="rect",x0=x0,x1=x1,y0=y0,y1=y1,fillcolor=fill,line_width=0,layer="below")
    for lbl,lx,ly in Q_ANN_P:
        fig_q.add_annotation(x=lx,y=ly,text=lbl,showarrow=False,font=dict(size=11,color="rgba(0,0,0,.18)"))
    fig_q.add_hline(y=company_monthly_avg,line_dash="dash",line_color="#607D8B",line_width=1.2,
                    annotation_text=f"2022全司月均 {company_monthly_avg:.1f}万",annotation_position="top right")
    fig_q.add_vline(x=exp_mid_wan,line_dash="dash",line_color="#607D8B",line_width=1.2,
                    annotation_text=f"部门费用中位 {exp_mid_wan:.1f}万",annotation_position="top left")

    for t in ["尖兵型","重炮型","潜伏型","失血型","待观察"]:
        fig_q.add_trace(go.Scatter(x=[None],y=[None],mode="markers",
            marker=dict(size=10,color=PERSON_COLORS[t]),name=t,showlegend=True))
    for _,row in df_quad.iterrows():
        is_zero=row["收入合计(元)"]==0
        pd_info=PTYPE_DESC_MAP.get(row["象限"],{})
        pd_sub=pd_info.get("sub","")
        pd_desc=pd_info.get("desc","")
        fig_q.add_trace(go.Scatter(
            x=[row["费用合计(万元)"]],y=[row["月均收入(万元)"]],
            mode="markers+text",
            marker=dict(size=18,color=PERSON_COLORS.get(row["象限"],"#607D8B"),
                        symbol="x" if is_zero else "circle",line=dict(color="white",width=1.5)),
            text=[row["市场人员"]],textposition="top center",textfont=dict(size=10),
            name=row["象限"],showlegend=False,
            hovertemplate=(
                f"<b>{row['市场人员']}【{row['象限']}】</b><br>"
                f"月均收入：{row['月均收入(万元)']:.2f}万<br>"
                f"累计费用：{row['费用合计(万元)']:.2f}万<br>"
                f"---<br>"
                f"类型分析：{pd_sub}<br>"
                f"{pd_desc}"
                f"<extra></extra>")))
    # 合作关系点
    ccolor=PERSON_COLORS.get(cq,"#607D8B")
    cq_info=PTYPE_DESC_MAP.get(cq,{})
    fig_q.add_trace(go.Scatter(
        x=[collab_exp_q],y=[collab_rev_q],mode="markers+text",
        marker=dict(size=22,color=ccolor,symbol="diamond",line=dict(color="white",width=2)),
        text=[collab_name],textposition="top center",textfont=dict(size=10),
        name=collab_name,showlegend=False,
        hovertemplate=(
            f"<b>{collab_name}【{cq}】</b><br>"
            f"人员构成：{cp1} 和 {cp2}<br>"
            f"月均收入：{collab_rev_q:.2f}万<br>"
            f"累计费用：{collab_exp_q:.2f}万<br>"
            f"---<br>"
            f"类型分析：{cq_info.get('sub','')}<br>"
            f"{cq_info.get('desc','')}"
            f"<extra></extra>")))
    fig_q.update_layout(chart_layout(480,0.6,{
        "xaxis_title":"差旅+招待费用合计（万元）","yaxis_title":"月均业绩（万元）",
        "legend":dict(title=dict(text="人员类型"),orientation="v",x=1.02,y=0.6,
                      bgcolor="rgba(255,255,255,.92)",bordercolor="#E0E0E0",borderwidth=1,itemsizing="constant"),
    }))

    st.plotly_chart(fig_q,use_container_width=True)

    st.markdown(
        '<div class="caption-box">统计口径：费用仅含差旅费及商务招待费；来源为报销申请明细和携程商旅记录；'
        '不含工资、奖金、社保；收入为实际开票金额；本司获客成本率基准 = 2%；费用上限 = 收入 × 2%；数据已脱敏处理。</div>',
        unsafe_allow_html=True)
