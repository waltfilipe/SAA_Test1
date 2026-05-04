import streamlit as st
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from mplsoccer import Pitch
import pandas as pd
import numpy as np
from PIL import Image
from io import BytesIO
from matplotlib.patches import FancyArrowPatch, Rectangle
from streamlit_image_coordinates import streamlit_image_coordinates
from matplotlib.colors import Normalize, LinearSegmentedColormap
from collections import defaultdict
import math
from scipy.ndimage import gaussian_filter

st.set_page_config(layout="wide", page_title="Action Map - Caleb Simmons")

st.markdown('''
<style>
.small-metric{padding:6px 8px;}
.small-metric .label{font-size:12px;color:#ffffff;margin-bottom:3px;opacity:.95;}
.small-metric .value{font-size:18px;font-weight:600;color:#ffffff;}
.small-metric .delta{font-size:11px;color:#e6e6e6;margin-top:4px;}
.stats-section-title{font-size:13px;font-weight:600;margin-bottom:6px;color:#ffffff;}
.streamlit-expanderHeader{color:#ffffff!important;}
.streamlit-expander{background:rgba(255,255,255,.02);}
.filter-panel{
  background:linear-gradient(168deg,rgba(30,41,59,.92) 0%,rgba(15,23,42,.97) 100%);
  border:1px solid rgba(255,255,255,.08);border-radius:14px;
  padding:20px 14px 16px 14px;
  box-shadow:0 4px 24px rgba(0,0,0,.25),0 1px 4px rgba(0,0,0,.12);
  backdrop-filter:blur(6px);}
.filter-panel h3{font-size:14px;color:#c8d6e5;letter-spacing:.5px;margin-bottom:8px;}
.filter-panel .filter-divider{border:none;border-top:1px solid rgba(255,255,255,.07);margin:12px 0;}
.stat-box{
  background:linear-gradient(145deg,rgba(30,41,59,.92) 0%,rgba(15,23,42,.97) 100%);
  border:1px solid rgba(255,255,255,.09);
  border-radius:12px;
  padding:12px 14px;
  margin-bottom:8px;
  box-shadow:0 3px 12px rgba(0,0,0,.3);}
.stat-box-accent-green{border-left:3px solid #10b981;}
.stat-box-accent-blue{border-left:3px solid #3b82f6;}
.stat-box-accent-amber{border-left:3px solid #f59e0b;}
.stat-box-accent-purple{border-left:3px solid #8b5cf6;}
.stat-box-accent-red{border-left:3px solid #ef4444;}
.stat-box-accent-cyan{border-left:3px solid #06b6d4;}
.stat-box .sb-label{font-size:10px;color:#94a3b8;letter-spacing:.7px;text-transform:uppercase;font-weight:600;}
.stat-box .sb-value{font-size:22px;font-weight:700;color:#f1f5f9;line-height:1.15;margin-top:2px;}
.stat-box .sb-sub{font-size:10px;color:#64748b;margin-top:3px;}
.section-title {font-size:22px; font-weight:800; color:#ffffff; margin:0 0 16px 0; padding-bottom:6px; border-bottom:1px solid rgba(255,255,255,0.1); letter-spacing:0.5px;}
</style>
''', unsafe_allow_html=True)

def stat_box(label, value, sub=None, accent='blue', delta=None, delta_good_is_up=True):
    html = f'<div class="stat-box stat-box-accent-{accent}">'
    html += f'<div class="sb-label">{label}</div>'
    html += f'<div style="display:flex; align-items:baseline; gap:8px;">'
    html += f'<div class="sb-value">{value}</div>'
    
    if delta is not None:
        if delta > 0:
            color = "#10b981" if delta_good_is_up else "#ef4444"
            arrow = "↑"
        elif delta < 0:
            color = "#ef4444" if delta_good_is_up else "#10b981"
            arrow = "↓"
        else:
            color = "#94a3b8"
            arrow = "−"
        
        # Formata delta mantendo o tipo (int ou float)
        delta_str = f"{abs(delta):.1f}" if isinstance(delta, float) else f"{abs(delta)}"
        html += f'<div style="font-size:13px; font-weight:700; color:{color};">{arrow} {delta_str}</div>'

    html += '</div>'
    if sub:
        html += f'<div class="sb-sub">{sub}</div>'
    html += '</div>'
    st.markdown(html, unsafe_allow_html=True)


# =============================================================================
# Constants
# =============================================================================
FIELD_X, FIELD_Y   = 120.0, 80.0
HALF_LINE_X        = FIELD_X / 2
FINAL_THIRD_LINE_X = 80.0
LANE_LEFT_MIN      = 53.33
LANE_RIGHT_MAX     = 26.67
NX, NY             = 16, 12
LATERAL_MIN_DIST   = 12.0
D_REF, D_SCALE, BONUS_CAP = 10.0, 20.0, 0.60

FIG_W, FIG_H = 9.2, 6.1
FIG_DPI      = 130

# Cores pastéis elegantes para as setas (Amarelo suave -> Laranja -> Vermelho rosado)
CMAP_TOP15 = LinearSegmentedColormap.from_list(
    "top15", ["#fde047", "#f59e0b", "#f43f5e", "#e11d48", "#be123c"]
)
NORM_TOP15 = Normalize(vmin=0.1, vmax=0.5)

# Cores mais claras e suaves para o Pass Density
CMAP_DENSITY = LinearSegmentedColormap.from_list(
    "density",
    ["#0f172a", "#1e1b4b", "#312e81", "#1d4ed8", "#0ea5e9",
     "#2dd4bf", "#ca8a04", "#f97316", "#fcd34d", "#fef08a"]
)

# =============================================================================
# xT grid
# =============================================================================
def distance_bonus(distance):
    excess = np.maximum(0.0, np.asarray(distance, dtype=float) - D_REF)
    return np.minimum(BONUS_CAP, np.log1p(excess / D_SCALE))

@st.cache_data(show_spinner=False)
def compute_xt_grid(NX=16, NY=12, sub=24,
    goal_width=11.0, penalty_depth=18.5, penalty_width=45.32,
    prox_w=0.50, central_w=0.50,
    internal_prox_power=2.8, internal_central_power=2.4, center_boost=0.20,
    FUNNEL_INFLUENCE_RANGE=35.0, FUNNEL_POWER=1.3, BASE_BOOST_WEIGHT=0.15,
    band_width_m=180.0, blur_window_m=60.0, final_blur_m=12.0,
    ANGLE_WEIGHT=0.50, ANGLE_POWER=1.4, BASE_ANGLE_WEIGHT=0.40):

    ncols_hr = NX * sub; nrows_hr = NY * sub
    xe = np.linspace(0, FIELD_X, ncols_hr+1); ye = np.linspace(0, FIELD_Y, nrows_hr+1)
    xc = (xe[:-1]+xe[1:])/2; yc_arr = (ye[:-1]+ye[1:])/2
    Xc, Yc = np.meshgrid(xc, yc_arr)
    xp = 0.01+(Xc/FIELD_X)*0.99; yc = 1.0-np.abs((Yc/FIELD_Y)-0.5)*2.0
    BASE = xp*(0.8+0.2*yc); BASE = (BASE-BASE.min())/(BASE.max()-BASE.min()+1e-12)
    cy = FIELD_Y/2.0
    fv = [(FIELD_X,cy-goal_width/2),(FIELD_X-penalty_depth,cy-penalty_width/2),
          (FIELD_X-penalty_depth,cy+penalty_width/2),(FIELD_X,cy+goal_width/2)]
    bpts = []
    for i in range(len(fv)):
        a,b = fv[i],fv[(i+1)%len(fv)]; dx,dy = b[0]-a[0],b[1]-a[1]
        n = max(2,int(round(math.hypot(dx,dy)/0.5)))
        for t in np.linspace(0,1,n,endpoint=False): bpts.append((a[0]+dx*t,a[1]+dy*t))
    bpts = np.array(bpts)
    fX=Xc.ravel(); fY=Yc.ravel(); md2=np.full(fX.size,np.inf)
    for bp in bpts: np.minimum(md2,(fX-bp[0])**2+(fY-bp[1])**2,out=md2)
    adist = np.sqrt(md2).reshape(Xc.shape)
    infl = np.clip((1-np.clip(adist/FUNNEL_INFLUENCE_RANGE,0,1))**FUNNEL_POWER,0,1)
    D = np.hypot(FIELD_X-Xc,cy-Yc)
    prox = 1-np.clip(D/np.hypot(FIELD_X,FIELD_Y/2),0,1)
    cent = 1-np.clip(np.abs((Yc-cy)/cy),0,1)
    ub = np.clip((prox_w*np.clip(prox**internal_prox_power,0,1)+
                  central_w*np.clip(cent**internal_central_power,0,1))*(1+center_boost*prox),0,1)
    v1x=FIELD_X-Xc; v1y=(cy+goal_width/2)-Yc; v2x=FIELD_X-Xc; v2y=(cy-goal_width/2)-Yc
    ca=np.clip((v1x*v2x+v1y*v2y)/(np.hypot(v1x,v1y)*np.hypot(v2x,v2y)+1e-12),-1,1)
    ang=np.arccos(ca); af=np.clip((ang/(ang.max()+1e-12))**ANGLE_POWER,0,1)
    ub=np.clip(ub*((1-ANGLE_WEIGHT)+ANGLE_WEIGHT*af),0,1)
    Bc=BASE*((1-BASE_ANGLE_WEIGHT)+BASE_ANGLE_WEIGHT*af)
    Bc=(Bc-Bc.min())/(Bc.max()-Bc.min()+1e-12); XTB=Bc+infl*BASE_BOOST_WEIGHT*ub
    pw=FIELD_X/ncols_hr; ph=FIELD_Y/nrows_hr
    rx=max(1,int(round((blur_window_m/pw)/2))); ry=max(1,int(round((blur_window_m/ph)/2)))
    def blur(a,rx,ry):
        H,W=a.shape; p=np.pad(a,((ry,ry),(rx,rx)),mode='edge').astype(np.float64)
        ii=p.cumsum(0).cumsum(1); s=ii[2*ry:2*ry+H,2*rx:2*rx+W].copy()
        s+=ii[:H,:W]; s-=ii[:H,2*rx:2*rx+W]; s-=ii[2*ry:2*ry+H,:W]
        return s/((2*ry+1)*(2*rx+1))
    w=0.5*(1-np.cos(np.pi*np.clip(adist/band_width_m,0,1)))
    XTbl=w*XTB+(1-w)*blur(XTB,rx,ry)
    rf=max(1,int(round((final_blur_m/pw)/2))); rfy=max(1,int(round((final_blur_m/ph)/2)))
    XT=0.85*XTbl+0.15*blur(XTbl,rf,rfy); XT=(XT-XT.min())/(XT.max()-XT.min()+1e-12)
    XTc=np.zeros((NY,NX))
    for iy in range(NY):
        for ix in range(NX): XTc[iy,ix]=XT[iy*sub:(iy+1)*sub,ix*sub:(ix+1)*sub].mean()
    XTc=(XTc-XTc.min())/(XTc.max()-XTc.min()+1e-12)
    return XTc, XT

XT_GRID, _ = compute_xt_grid()

def xt_value(x, y):
    ix = int(np.clip((x/FIELD_X)*NX, 0, NX-1))
    iy = int(np.clip((y/FIELD_Y)*NY, 0, NY-1))
    return float(XT_GRID[iy, ix])

# =============================================================================
# Match data
# =============================================================================
matches_data = {
    "Vs Connecticut": [
        ('ACTION WON',26.75,68.34,8.97,51.05,None),('ACTION WON',31.24,51.22,34.57,72.50,None),
        ('ACTION WON',36.06,46.90,44.37,57.04,None),('ACTION WON',48.36,64.02,58.17,51.72,None),
        ('ACTION WON',58.17,64.02,62.49,55.21,None),('ACTION WON',54.51,49.72,64.82,61.69,None),
        ('ACTION WON',42.21,70.84,34.90,76.49,None),('ACTION WON',43.54,75.32,36.73,67.84,None),
        ('ACTION WON',32.24,53.96,6.81,38.50,None),('ACTION WON',33.57,65.77,36.56,75.57,None),
        ('ACTION WON',37.39,61.11,43.04,75.41,None),('ACTION WON',65.49,53.63,56.18,70.42,None),
        ('ACTION WON',55.68,48.15,46.87,30.86,None),('ACTION WON',52.02,22.05,46.70,41.99,None),
        ('ACTION WON',62.16,35.51,71.80,35.18,None),('ACTION WON',54.02,33.35,63.99,22.55,None),
        ('ACTION WON',60.00,22.21,76.62,32.85,None),('ACTION WON',87.10,9.41,77.45,16.23,None),
        ('ACTION WON',62.66,20.05,117.18,8.25,None),('ACTION WON',98.90,43.49,103.22,47.15,None),
        ('ACTION WON',70.31,45.98,82.28,60.11,None),('ACTION WON',85.10,75.24,101.39,74.08,None),
        ('ACTION WON',53.18,67.59,39.05,59.62,None),('ACTION WON',55.18,49.64,54.85,13.07,None),
        ('ACTION WON',68.64,19.22,49.03,24.37,None),('ACTION WON',53.35,22.71,59.34,30.19,None),
        ('ACTION WON',44.37,24.71,40.05,46.82,None),('ACTION WON',43.88,39.34,41.38,73.08,None),
        ('ACTION WON',56.84,53.46,70.81,76.24,None),('ACTION WON',82.77,12.24,91.42,4.59,None),
        ('ACTION WON',108.04,11.74,115.69,58.29,None),('ACTION WON',93.08,3.93,111.03,13.74,None),
        ('ACTION WON',84.60,17.89,96.74,22.05,None),('ACTION WON',58.34,16.06,65.65,2.43,None),
        ('ACTION WON',52.02,8.58,44.37,15.73,None),('ACTION WON',61.00,23.21,49.36,15.23,None),
        ('ACTION WON',32.74,30.69,50.03,33.02,None),('ACTION WON',51.85,33.68,60.66,40.00,None),
        ('ACTION WON',79.95,60.45,98.23,60.28,None),('ACTION WON',31.24,52.14,39.05,72.08,None),
        ('ACTION WON',39.72,48.98,33.40,57.62,None),('ACTION WON',70.64,51.47,61.00,51.64,None),
        ('ACTION LOST',53.35,19.55,73.96,11.24,None),('ACTION LOST',63.82,20.55,88.76,22.55,None),
        ('ACTION LOST',85.60,27.86,94.41,37.17,None),('ACTION LOST',77.79,27.53,96.41,25.37,None),
        ('ACTION LOST',91.09,27.86,109.54,50.47,None),('ACTION LOST',58.17,26.04,95.41,40.33,None),
        ('ACTION LOST',53.35,28.53,73.80,27.86,None),('ACTION LOST',53.35,34.02,84.60,58.62,None),
        ('ACTION LOST',56.18,49.48,97.07,62.11,None),('ACTION LOST',34.23,74.91,65.65,78.57,None),
    ],
    "Vs Nashville": [
        ('ACTION WON',21.27,14.23,29.25,31.02,None),('ACTION WON',29.41,23.38,34.40,64.60,None),
        ('ACTION WON',41.55,39.67,41.88,6.92,None),('ACTION WON',44.54,32.52,43.54,14.23,None),
        ('ACTION WON',23.59,56.46,34.57,47.48,None),('ACTION WON',30.58,64.44,21.10,49.48,None),
        ('ACTION WON',33.07,56.79,49.53,69.59,None),('ACTION WON',33.24,59.78,44.04,71.75,None),
        ('ACTION WON',61.50,71.58,54.68,75.57,None),('ACTION WON',63.16,50.81,78.45,67.26,None),
        ('ACTION WON',63.49,76.90,84.44,62.77,None),('ACTION WON',76.96,56.96,86.93,57.79,None),
        ('ACTION WON',82.61,59.12,96.41,68.43,None),('ACTION WON',79.78,35.35,106.21,11.74,None),
        ('ACTION WON',45.37,49.64,40.72,32.02,None),
        ('ACTION LOST',78.62,64.94,96.57,67.10,None),('ACTION LOST',85.43,68.76,106.05,77.74,None),
    ],
    "Vs Seongnam": [
        ('ACTION WON',28.08,28.53,29.75,8.25,None),('ACTION WON',33.74,26.54,29.41,43.82,None),
        ('ACTION WON',28.08,47.15,31.57,64.60,None),('ACTION WON',39.39,43.82,51.69,53.46,None),
        ('ACTION WON',43.88,46.15,55.84,40.66,None),('ACTION WON',47.03,49.97,44.04,28.03,None),
        ('ACTION WON',47.53,50.81,71.97,33.18,None),('ACTION WON',67.65,52.63,64.32,33.85,None),
        ('ACTION WON',73.63,65.10,69.31,73.25,None),('ACTION WON',77.29,63.27,79.12,72.91,None),
        ('ACTION WON',81.61,56.62,93.91,73.75,None),('ACTION WON',86.43,66.43,81.78,54.96,None),
        ('ACTION WON',111.03,71.42,99.56,67.59,None),('ACTION WON',89.76,59.62,97.74,48.98,None),
        ('ACTION WON',88.43,52.47,96.41,74.24,None),('ACTION WON',87.93,50.97,77.12,27.70,None),
        ('ACTION WON',81.61,53.63,74.30,27.03,None),('ACTION WON',79.28,51.14,94.91,70.42,None),
        ('ACTION WON',52.85,32.85,65.49,25.37,None),('ACTION WON',82.77,33.18,69.31,47.65,None),
        ('ACTION LOST',72.14,16.56,78.45,1.60,None),('ACTION LOST',79.62,27.53,97.07,47.98,None),
        ('ACTION LOST',91.75,50.14,109.70,65.77,None),('ACTION LOST',96.41,56.79,107.04,67.26,None),
    ],
    "Vs Red Bull": [
        ('ACTION WON',39.39,19.39,52.35,4.76,None),('ACTION WON',63.82,7.92,72.63,1.43,None),
        ('ACTION WON',70.47,11.91,80.95,13.74,None),('ACTION WON',64.49,22.55,97.24,10.24,None),
        ('ACTION WON',32.07,35.51,43.04,28.20,None),('ACTION WON',53.52,46.32,54.02,33.68,None),
        ('ACTION WON',77.12,48.64,84.94,50.14,None),('ACTION WON',78.12,52.47,117.52,69.42,None),
        ('ACTION WON',88.76,65.93,97.40,76.74,None),('ACTION WON',82.61,69.26,86.60,77.40,None),
        ('ACTION WON',78.62,66.26,79.62,78.40,None),('ACTION WON',83.61,75.91,62.49,57.12,None),
        ('ACTION WON',34.40,50.14,88.76,75.41,None),('ACTION WON',56.68,64.27,78.29,64.27,None),
        ('ACTION WON',51.85,73.25,54.18,78.07,None),('ACTION WON',41.05,57.45,46.04,74.91,None),
        ('ACTION WON',37.39,60.61,41.71,73.91,None),('ACTION WON',30.41,63.44,36.89,77.40,None),
        ('ACTION WON',26.09,63.94,28.42,76.74,None),('ACTION WON',22.43,56.62,22.10,76.41,None),
        ('ACTION WON',33.90,64.77,25.42,73.58,None),
        ('ACTION LOST',41.88,42.49,56.18,52.97,None),('ACTION LOST',37.56,41.16,46.37,53.96,None),
        ('ACTION LOST',54.68,56.96,54.85,64.44,None),('ACTION LOST',51.69,68.43,66.15,76.57,None),
    ],
}

def classify_action_direction(x0, y0, x1, y1):
    dx,dy = x1-x0,y1-y0; dist=np.sqrt(dx**2+dy**2)
    ang=np.degrees(np.arctan2(abs(dy),dx))
    if ang<=45:  return 'forward'
    if ang>=135: return 'backward'
    return 'lateral' if dist>LATERAL_MIN_DIST else ('forward' if dx>=0 else 'backward')

def recompute_bonus(df):
    df=df.copy()
    excess=np.maximum(0.0,df['action_distance'].values-D_REF)
    df['dist_bonus']   =np.minimum(BONUS_CAP,np.log1p(excess/D_SCALE))
    df['delta_xt_adj'] =np.where(df['outcome']=='successful',df['delta_xt']*(1.0+df['dist_bonus']),0.0)
    return df

dfs_by_match = {}
for match_name, events in matches_data.items():
    dfm=pd.DataFrame(events,columns=['type','x_start','y_start','x_end','y_end','video'])
    dfm['match']=match_name; dfm['number']=np.arange(1,len(dfm)+1)
    dfm['is_won']=dfm['type'].str.contains('WON',case=False)
    dfm['outcome']=np.where(dfm['is_won'],'successful','failed')
    dfm['direction']=dfm.apply(lambda r:classify_action_direction(r.x_start,r.y_start,r.x_end,r.y_end),axis=1)
    dfm['is_forward']=dfm['direction']=='forward'; dfm['is_backward']=dfm['direction']=='backward'
    dfm['is_lateral']=dfm['direction']=='lateral'
    dfm['xt_start']=dfm.apply(lambda r:xt_value(r.x_start,r.y_start),axis=1)
    dfm['xt_end']  =dfm.apply(lambda r:xt_value(r.x_end,  r.y_end),  axis=1)
    dfm['delta_xt']=np.where(dfm['outcome']=='successful',dfm['xt_end']-dfm['xt_start'],0.0)
    dfm['action_distance']=np.sqrt((dfm.x_end-dfm.x_start)**2+(dfm.y_end-dfm.y_start)**2)
    dfm['dist_bonus']=distance_bonus(dfm['action_distance'].values)
    dfm['delta_xt_adj']=np.where(dfm['outcome']=='successful',dfm['delta_xt']*(1+dfm['dist_bonus']),0.0)
    dfs_by_match[match_name]=dfm

df_all=pd.concat(dfs_by_match.values(),ignore_index=True)
full_data={'All Matches':df_all}; full_data.update(dfs_by_match)

# =============================================================================
# Stats & Aggregation
# =============================================================================
def compute_stats(df):
    total=len(df); successful=int(df['is_won'].sum())
    accuracy=(successful/total*100) if total else 0.0
    succ_mask=df['outcome']=='successful'
    sum_dxt=float(df.loc[succ_mask,'delta_xt_adj'].sum()) if succ_mask.any() else 0.0
    pos_mask=succ_mask&(df['delta_xt_adj']>0); pos_count=int(pos_mask.sum())
    pos_mean=float(df.loc[pos_mask,'delta_xt_adj'].mean()) if pos_count else 0.0
    pos_pct=(pos_count/total*100) if total else 0.0
    
    high_xt_count = int((df['delta_xt_adj'] > 0.05).sum())
    high_xt_pct = (high_xt_count / total * 100) if total else 0.0
    
    top15_df=df.loc[pos_mask].sort_values('delta_xt_adj',ascending=False).head(15)
    top15_sum =float(top15_df['delta_xt_adj'].sum())  if not top15_df.empty else 0.0
    top15_mean=float(top15_df['delta_xt_adj'].mean()) if not top15_df.empty else 0.0
    xt_end_sum =float(df.loc[succ_mask,'xt_end'].sum())  if succ_mask.any() else 0.0
    xt_end_mean=float(df.loc[succ_mask,'xt_end'].mean()) if succ_mask.any() else 0.0
    fail_mask=df['outcome']=='failed'; fail_count=int(fail_mask.sum())
    fail_xt_sum=float((1.0-df.loc[fail_mask,'xt_end']).sum()) if fail_count else 0.0
    
    return {
        'total':total,'successful':successful,'failed':fail_count,
        'accuracy':round(accuracy,2),
        'sum_dxt':round(sum_dxt,4),'pos_pct':round(pos_pct,2),'pos_mean':round(pos_mean,4),
        'high_xt_pct': round(high_xt_pct, 2),
        'top15_sum':round(top15_sum,4),'top15_mean':round(top15_mean,4),
        'xt_end_sum':round(xt_end_sum,4),'xt_end_mean':round(xt_end_mean,4),
        'fail_xt_sum':round(fail_xt_sum,4),
        'fwd':int(df['is_forward'].sum()),'bwd':int(df['is_backward'].sum()),'lat':int(df['is_lateral'].sum()),
    }

# Calcular médias globais para o delta no painel
all_match_stats = [compute_stats(recompute_bonus(full_data[k])) for k in matches_data.keys()]
avg_stats = {}
if all_match_stats:
    for k in all_match_stats[0].keys():
        avg_stats[k] = sum(m[k] for m in all_match_stats) / len(all_match_stats)

# =============================================================================
# Drawing helpers
# =============================================================================
def _base_pitch(bg='#1a1a2e', line_alpha=0.95, line_zorder=1.2, solid_lines=False):
    linestyle = '-' if solid_lines else '-'
    pitch = Pitch(pitch_type='statsbomb', pitch_color=bg,
                  line_color='#ffffff', line_alpha=line_alpha, line_zorder=line_zorder, linestyle=linestyle)
    fig, ax = pitch.draw(figsize=(FIG_W, FIG_H))
    fig.set_facecolor(bg); fig.set_dpi(FIG_DPI)
    fig.subplots_adjust(bottom=0.10) # Space for attack arrow
    return fig, ax, pitch

def _attack_arrow(fig, ax, has_cbar=False):
    ap = ax.get_position()
    cx = (ap.x0 + ap.x1) / 2
    if has_cbar:
        cx -= 0.02 # Desloca ligeiramente para compensar a barra de gradiente
    sm = ap.y0 - 0.04
    fig.patches.append(FancyArrowPatch(
        (cx-0.055,sm),(cx+0.055,sm),transform=fig.transFigure,
        arrowstyle='-|>',mutation_scale=13,linewidth=1.8,color='#cccccc'))
    fig.text(cx,sm-0.007,'Attack Direction',ha='center',va='top',
             transform=fig.transFigure,fontsize=8.5,color='#cccccc')

def _save_fig(fig):
    fig.canvas.draw()
    buf=BytesIO()
    fig.savefig(buf,format='png',dpi=FIG_DPI,facecolor=fig.get_facecolor(),bbox_inches='tight')
    buf.seek(0); return Image.open(buf)

def _draw_soft_arrow(pitch, ax, x0, y0, x1, y1, color, is_sel=False):
    width = 2.2; hw = 4.5; hl = 4.5; alpha = 0.95
    if is_sel:
        color = '#00f0ff'; alpha = 1.0; width = 3.5; hw = 5.5; hl = 5.5
        
    pitch.arrows(x0, y0, x1, y1,
                 color=color, width=width, headwidth=hw, headlength=hl,
                 ax=ax, zorder=4, alpha=alpha)
    dot_s = 35 if is_sel else 25
    pitch.scatter(x0, y0, s=dot_s, marker='o', color=color,
                  edgecolors='white', linewidths=0.6,
                  ax=ax, zorder=6, alpha=alpha)

# =============================================================================
# Top-15 map
# =============================================================================
def draw_top15_map(df, title, selected_num=None):
    fig, ax, pitch = _base_pitch()
    
    ax.axvline(x=FINAL_THIRD_LINE_X, color='#ffffff', lw=1.15, alpha=0.15, linestyle='--')
    ax.axvline(x=HALF_LINE_X,        color='#ffffff',  lw=0.6,  alpha=0.10, linestyle='--')

    top15 = (
        df[(df['outcome']=='successful')&(df['delta_xt_adj']>0)]
        .sort_values('delta_xt_adj',ascending=False).head(15)
        .copy().reset_index(drop=True)
    )
    top15['rank'] = np.arange(1, len(top15)+1)

    if not top15.empty:
        for _, row in top15.iterrows():
            val   = float(row['delta_xt_adj'])
            color = CMAP_TOP15(NORM_TOP15(np.clip(val, 0.1, 0.5)))
            is_sel = selected_num is not None and int(row['number'])==selected_num
            _draw_soft_arrow(pitch, ax,
                             float(row.x_start), float(row.y_start),
                             float(row.x_end),   float(row.y_end),
                             color, is_sel)
    
    sm = plt.cm.ScalarMappable(cmap=CMAP_TOP15, norm=NORM_TOP15)
    cbar = fig.colorbar(sm, ax=ax, fraction=0.025, pad=0.03, shrink=0.65)
    cbar.set_label('ΔxT', color='#ffffff', fontsize=9)
    cbar.ax.yaxis.set_tick_params(color='#ffffff', labelsize=8)
    plt.setp(plt.getp(cbar.ax.axes, 'yticklabels'), color='#ffffff')

    _attack_arrow(fig, ax, has_cbar=True)
    return _save_fig(fig), ax, fig, top15

# =============================================================================
# Zone corridor heatmap
# =============================================================================
def _zone_bins():
    return np.linspace(0, FIELD_X, 7), np.array([0.0, LANE_RIGHT_MAX, LANE_LEFT_MIN, FIELD_Y])

def draw_corridor_heatmap(df):
    df_s=df[df['is_won']].copy()
    x_bins,_=_zone_bins()
    corridors={'left':(LANE_LEFT_MIN,FIELD_Y),'center':(LANE_RIGHT_MAX,LANE_LEFT_MIN),'right':(0.0,LANE_RIGHT_MAX)}
    counts={}
    for cname,(y0,y1) in corridors.items():
        arr=np.zeros(6,dtype=int)
        for i in range(6):
            x0_,x1_=x_bins[i],x_bins[i+1]
            arr[i]=int(((df_s['x_end']>=x0_)&(df_s['x_end']<x1_)
                        &(df_s['y_end']>=y0)&(df_s['y_end']<y1)).sum())
        counts[cname]=arr
    all_vals=np.concatenate([counts[c] for c in counts])
    vmax=max(1,int(all_vals.max()))
    cmap=LinearSegmentedColormap.from_list("wr",["#ffffff","#ffecec","#ffbfbf","#ff8080","#ff3b3b","#ff0000"])
    norm=Normalize(vmin=0,vmax=vmax); thr=max(1,vmax*0.35)

    fig, ax, pitch = _base_pitch()
    for cname,(y0,y1) in corridors.items():
        for i in range(6):
            x0_,x1_=x_bins[i],x_bins[i+1]; value=counts[cname][i]
            if value == 0: continue
            ax.add_patch(Rectangle((x0_,y0),x1_-x0_,y1-y0,
                                   facecolor=cmap(norm(value)),edgecolor=(1,1,1,0.12),lw=0.6,alpha=0.95,zorder=2))
            ax.text((x0_+x1_)/2,(y0+y1)/2,str(value),ha='center',va='center',
                    color='#000000' if value<=thr else '#ffffff',
                    fontsize=10,fontweight='700' if value>=vmax*0.5 else '600',zorder=4)
    ax.axhline(y=LANE_LEFT_MIN, color='#ffffff',lw=0.5,alpha=0.15,linestyle='--',zorder=3)
    ax.axhline(y=LANE_RIGHT_MAX,color='#ffffff',lw=0.5,alpha=0.15,linestyle='--',zorder=3)
    _attack_arrow(fig, ax, has_cbar=False)
    return _save_fig(fig), ax, fig

# =============================================================================
# Density heatmap
# =============================================================================
def draw_density_heatmap(df):
    NX_D, NY_D = 30, 20
    x_edges = np.linspace(0, FIELD_X, NX_D+1)
    y_edges = np.linspace(0, FIELD_Y, NY_D+1)

    all_x = np.concatenate([df['x_start'].values, df['x_end'].values])
    all_y = np.concatenate([df['y_start'].values, df['y_end'].values])
    counts, _, _ = np.histogram2d(all_x, all_y, bins=[x_edges, y_edges])
    counts = gaussian_filter(counts.T.astype(float), sigma=0.85)

    # Linhas do campo sólidas e em primeiro plano (zorder elevado)
    fig, ax, pitch = _base_pitch(bg='#0f172a', line_zorder=5, solid_lines=True)

    vmax = max(1.0, counts.max())
    norm = Normalize(vmin=0, vmax=vmax)

    margin = 0.28   
    for iy in range(NY_D):
        for ix in range(NX_D):
            val = counts[iy, ix]
            face_c = CMAP_DENSITY(norm(val)) if val >= 0.04 else '#0f172a'
            x0_ = x_edges[ix]   + margin
            x1_ = x_edges[ix+1] - margin
            y0_ = y_edges[iy]   + margin
            y1_ = y_edges[iy+1] - margin
            ax.add_patch(Rectangle(
                (x0_, y0_), x1_-x0_, y1_-y0_,
                facecolor=face_c,
                edgecolor='white',
                lw=0.4,
                alpha=0.88, zorder=2
            ))

    sm=plt.cm.ScalarMappable(cmap=CMAP_DENSITY,norm=norm)
    cbar=fig.colorbar(sm,ax=ax,fraction=0.016,pad=0.02,shrink=0.70)
    cbar.set_label('Density',color='#a0b4c8',fontsize=8,labelpad=2)
    cbar.ax.yaxis.set_tick_params(color='#a0b4c8',labelsize=6)
    plt.setp(plt.getp(cbar.ax.axes,'yticklabels'),color='#a0b4c8')
    _attack_arrow(fig, ax, has_cbar=True)
    return _save_fig(fig), ax, fig

# =============================================================================
# Session state
# =============================================================================
for key,default in [('selected_action',None),('last_map_click',None),('last_match',None)]:
    if key not in st.session_state: st.session_state[key]=default

# =============================================================================
# TABS
# =============================================================================
tab_map, tab_analysis = st.tabs(['📍 Map', '📊 Stats'])

# =============================================================================
# TAB MAP
# =============================================================================
with tab_map:
    col_f, col_map, col_stats = st.columns([0.72, 2.05, 1.08], gap='medium')

    with col_f:
        st.markdown('<div class="filter-panel">', unsafe_allow_html=True)
        st.markdown('### 🏟 Match')
        selected_match=st.selectbox('Choose match',list(full_data.keys()),index=0,label_visibility='collapsed')
        st.markdown('<hr class="filter-divider">', unsafe_allow_html=True)
        st.markdown('### ℹ️ Info')
        st.caption('Top 15 passes by ΔxT.\nClick an arrow origin dot to inspect.')
        st.markdown('</div>', unsafe_allow_html=True)

    if st.session_state['last_match']!=selected_match:
        st.session_state['selected_action']=None
        st.session_state['last_map_click']=None
        st.session_state['last_match']=selected_match

    df_base=recompute_bonus(full_data[selected_match].copy())
    s=compute_stats(df_base)

    with col_map:
        cur_sel=st.session_state.get('selected_action')
        sel_num=int(cur_sel['number']) if cur_sel is not None else None

        st.markdown('<div class="section-title">Top Actions Map</div>', unsafe_allow_html=True)
        img_obj,ax,fig,top15_df=draw_top15_map(df_base,title="",selected_num=sel_num)

        MAP_W=880
        click=streamlit_image_coordinates(img_obj,width=MAP_W,key='map_img')
        if click is not None and click!=st.session_state['last_map_click']:
            st.session_state['last_map_click']=click
            if not top15_df.empty:
                rw,rh=img_obj.size
                px=click['x']*(rw/click['width']); py=click['y']*(rh/click['height'])
                fx,fy=ax.transData.inverted().transform((px,rh-py))
                tmp=top15_df.copy()
                tmp['_d']=np.sqrt((tmp.x_start-fx)**2+(tmp.y_start-fy)**2)
                cands=tmp[tmp['_d']<6.0].sort_values('_d')
                if not cands.empty: st.session_state['selected_action']=cands.iloc[0]
        plt.close(fig)

        st.markdown("<br>", unsafe_allow_html=True)
        heatmap_mode = st.radio("Select View", ["Zone Heatmap", "Pass Density"], horizontal=True, label_visibility="collapsed")
        
        if heatmap_mode == "Zone Heatmap":
            zh_img,_,zh_fig=draw_corridor_heatmap(df_base)
            st.image(zh_img,use_container_width=True); plt.close(zh_fig)
        else:
            dh_img,_,dh_fig=draw_density_heatmap(df_base)
            st.image(dh_img,use_container_width=True); plt.close(dh_fig)

    with col_stats:
        
        # Exibimos delta de variação com a média apenas se não for "All Matches"
        is_single_match = (selected_match != 'All Matches')
        
        d_acc = (s['accuracy'] - avg_stats['accuracy']) if is_single_match else None
        stat_box('Accuracy', f"{s['accuracy']:.1f}%", f"{s['successful']} successful / {s['total']} total", accent='green', delta=d_acc)
        
        d_dxt = (s['sum_dxt'] - avg_stats['sum_dxt']) if is_single_match else None
        stat_box('Σ ΔxT', f"{s['sum_dxt']:.3f}", f"Avg positive: {s['pos_mean']:.3f}", accent='amber', delta=d_dxt)
        
        d_pospct = (s['pos_pct'] - avg_stats['pos_pct']) if is_single_match else None
        stat_box('% Positive ΔxT', f"{s['pos_pct']:.1f}%", f"Positive count: {s['successful']}", accent='blue', delta=d_pospct)

        d_highpct = (s['high_xt_pct'] - avg_stats['high_xt_pct']) if is_single_match else None
        stat_box('% ΔxT > 0.05', f"{s['high_xt_pct']:.1f}%", "Actions generating high threat", accent='cyan', delta=d_highpct)

        c1,c2=st.columns(2)
        with c1: stat_box('Σ End xT', f"{s['xt_end_sum']:.2f}", accent='purple', 
                          delta=(s['xt_end_sum']-avg_stats['xt_end_sum']) if is_single_match else None)
        with c2: stat_box('Σ xT Failed', f"{s['fail_xt_sum']:.2f}", accent='red', 
                          delta=(s['fail_xt_sum']-avg_stats['fail_xt_sum']) if is_single_match else None, delta_good_is_up=False)

        st.markdown(
            f'<div style="display:grid;grid-template-columns:repeat(3,1fr);gap:4px;margin-bottom:8px;">'
            f'<div class="stat-box" style="text-align:center;">'
            f'<div class="sb-label">Forward</div><div class="sb-value" style="font-size:17px;">{s["fwd"]}</div></div>'
            f'<div class="stat-box" style="text-align:center;">'
            f'<div class="sb-label">Backward</div><div class="sb-value" style="font-size:17px;">{s["bwd"]}</div></div>'
            f'<div class="stat-box" style="text-align:center;">'
            f'<div class="sb-label">Lateral</div><div class="sb-value" style="font-size:17px;">{s["lat"]}</div></div>'
            f'</div>', unsafe_allow_html=True)

        st.markdown('<h4 style="color:#ffffff;margin:16px 0 3px 0;">Event Panel</h4>',unsafe_allow_html=True)
        sel=st.session_state.get('selected_action')
        if sel is None:
            st.info('Click an origin dot on the map to inspect.')
        else:
            rank_v=int(sel['rank']) if 'rank' in sel.index and not pd.isna(sel['rank']) else None
            hex_c=matplotlib.colors.to_hex(CMAP_TOP15(NORM_TOP15(np.clip(float(sel['delta_xt_adj']),0.1,0.5))))
            st.markdown(
                f'<div style="display:flex;align-items:center;gap:9px;margin-bottom:7px;">'
                f'<span style="display:inline-block;width:12px;height:12px;border-radius:50%;'
                f'background:{hex_c};border:2px solid #fff;"></span>'
                f'<strong style="color:#fff;font-size:13px;">Action #{int(sel["number"])}'
                f'{" | Rank #"+str(rank_v) if rank_v else ""}</strong></div>',
                unsafe_allow_html=True)
            ca,cb=st.columns(2)
            with ca:
                st.write(f'**Start:** ({sel["x_start"]:.2f}, {sel["y_start"]:.2f})')
                st.write(f'**End:** ({sel["x_end"]:.2f}, {sel["y_end"]:.2f})')
                st.write(f'**Dir:** {str(sel["direction"]).capitalize()}')
            with cb:
                st.metric('Distance',f'{float(sel["action_distance"]):.1f} m')
                st.metric('ΔxT',f'{float(sel["delta_xt_adj"]):.4f}')
            if pd.notna(sel['video']) and str(sel['video']).strip()!='':
                try: st.video(sel['video'])
                except: st.error('Video not found.')
            else:
                st.caption('No video attached.')

# =============================================================================
# TAB ANALYSES
# =============================================================================
with tab_analysis:
    st.markdown('<div class="section-title">Stats</div>', unsafe_allow_html=True)
    sel_an=st.selectbox('Match',list(full_data.keys()),index=0,key='an_match')
    df_an=recompute_bonus(full_data[sel_an].copy()); s_an=compute_stats(df_an)
    is_an_single = (sel_an != 'All Matches')

    k1,k2,k3,k4=st.columns(4)
    
    def _kpi(col,label,value,sub,border, delta=None, delta_good_is_up=True):
        delta_html = ""
        if delta is not None:
            if delta > 0:
                c_delta = "#10b981" if delta_good_is_up else "#ef4444"
                arr = "↑"
            elif delta < 0:
                c_delta = "#ef4444" if delta_good_is_up else "#10b981"
                arr = "↓"
            else:
                c_delta = "#94a3b8"
                arr = "−"
            delta_html = f'<span style="font-size:14px; font-weight:700; color:{c_delta}; margin-left:10px;">{arr} {abs(delta):.1f}</span>'

        col.markdown(
            f'<div style="background:rgba(255,255,255,.04);border-left:4px solid {border};'
            f'border-radius:11px;padding:14px 16px; margin-bottom:12px;">'
            f'<div style="font-size:10px;color:#94a3b8;letter-spacing:.7px;text-transform:uppercase;font-weight:600;">{label}</div>'
            f'<div style="display:flex; align-items:baseline;">'
            f'<div style="font-size:28px;font-weight:700;color:#ffffff;line-height:1.15;">{value}</div>'
            f'{delta_html}'
            f'</div>'
            f'<div style="font-size:10px;color:#64748b;margin-top:4px;">{sub}</div></div>',
            unsafe_allow_html=True)

    _kpi(k1,'Accuracy', f"{s_an['accuracy']:.1f}%", f"{s_an['successful']} / {s_an['total']} successful",'#10b981', 
         delta=(s_an['accuracy']-avg_stats['accuracy']) if is_an_single else None)
    _kpi(k2,'Σ ΔxT', f"{s_an['sum_dxt']:.3f}", f"Avg positive ΔxT: {s_an['pos_mean']:.3f}",'#f59e0b', 
         delta=(s_an['sum_dxt']-avg_stats['sum_dxt']) if is_an_single else None)
    _kpi(k3,'% Positive ΔxT', f"{s_an['pos_pct']:.1f}%", f"Count w/ ΔxT > 0",'#3b82f6', 
         delta=(s_an['pos_pct']-avg_stats['pos_pct']) if is_an_single else None)
    _kpi(k4,'% ΔxT > 0.05', f"{s_an['high_xt_pct']:.1f}%", "Actions generating high threat",'#06b6d4', 
         delta=(s_an['high_xt_pct']-avg_stats['high_xt_pct']) if is_an_single else None)

    st.markdown('<div style="height:8px;"></div>',unsafe_allow_html=True)
    p1,p2,p3,p4=st.columns(4)
    with p1: small_metric('Σ Top 15 ΔxT',  f"{s_an['top15_sum']:.3f}")
    with p2: small_metric('Avg Top 15 ΔxT', f"{s_an['top15_mean']:.3f}")
    with p3: small_metric('Σ End xT',        f"{s_an['xt_end_sum']:.3f}")
    with p4: small_metric('Σ xT Failed',     f"{s_an['fail_xt_sum']:.3f}")

    st.markdown('<div style="height:4px;"></div>',unsafe_allow_html=True)
    d1,d2,d3=st.columns(3)
    with d1: small_metric('Forward',  f"{s_an['fwd']}", delta=f"{s_an['fwd']/max(s_an['total'],1)*100:.0f}% of total")
    with d2: small_metric('Backward', f"{s_an['bwd']}", delta=f"{s_an['bwd']/max(s_an['total'],1)*100:.0f}% of total")
    with d3: small_metric('Lateral',  f"{s_an['lat']}", delta=f"{s_an['lat']/max(s_an['total'],1)*100:.0f}% of total")
