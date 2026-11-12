#!/usr/bin/env python3
"""Render the article's data-derived figures from the archived records.

Usage: python scripts/render_paper_figures.py OUTPUT_DIR

Makes no model calls and runs no simulations: every input is read from data/
and its SHA256 is recorded in OUTPUT_DIR/figure_render_audit.json. Native PDF
width equals the article's 406.5 pt text width; Times New Roman must be
installed. References use the global KPI medoid, with each shift matched
separately. S3d is an uncertified anchor, never a deployed plan.
"""
from pathlib import Path
import hashlib
import json
import sys
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Patch, FancyBboxPatch
from matplotlib.text import Text
from matplotlib import font_manager
import subprocess

HERE = Path(__file__).resolve().parent
CASE = HERE.parent  # repository root
if len(sys.argv) != 2 or sys.argv[1] in ('-h', '--help'):
    sys.exit(__doc__)
PAPER = Path(sys.argv[1]).resolve()  # output directory
PAPER.mkdir(parents=True, exist_ok=True)
sys.path.insert(0, str(CASE / 'scripts'))
from paper_values import medoid, phases_by_canonical
DATA = CASE / 'data'
SINGLE = DATA / 'repeated/single_full_claude-sonnet-4-6_integrated_v5'
FACTORY = DATA / 'repeated/factory_claude-sonnet-4-6_analytical_v3'
W = 406.5 / 72
BLUE, ORANGE, GREEN, PURPLE = '#2476aa', '#cc7125', '#36804a', '#8655a1'
FONT_FAMILY = 'Times New Roman'
# Require the real font: do not silently substitute a similar typeface.
for weight in ('normal', 'bold'):
    font_manager.findfont(font_manager.FontProperties(family=FONT_FAMILY, weight=weight), fallback_to_default=False)
plt.rcParams.update({'font.family':FONT_FAMILY, 'font.size':9,
    'axes.labelsize':9, 'axes.titlesize':10, 'xtick.labelsize':9,
    'ytick.labelsize':9, 'legend.fontsize':9, 'pdf.fonttype':42,
    'axes.spines.top':False, 'axes.spines.right':False, 'axes.axisbelow':True})
REPORT = {'font_family':FONT_FAMILY, 'native_width_pt':406.5, 'inputs':{}, 'figures':{}, 'references':{},
 'corrections':['Reference bars now use batch KPI medoids and corresponding shifts.',
                'S1 Monte Carlo feasibility is displayed as 99.5%, not rounded to 100%.',
                'S3d is separately labelled as an uncertified anchor.']}

def read(path):
    REPORT['inputs'][str(path.relative_to(CASE))] = hashlib.sha256(path.read_bytes()).hexdigest()
    return json.loads(path.read_text())

def save(fig, name):
    fig.canvas.draw()
    renderer = fig.canvas.get_renderer()
    ignored = set()
    for ax in fig.axes:
        for axis in (ax.xaxis, ax.yaxis):
            lo, hi = sorted(axis.get_view_interval())
            for tick in axis.get_major_ticks() + axis.get_minor_ticks():
                if not lo <= tick.get_loc() <= hi:
                    ignored.update([tick.label1, tick.label2])
    texts = [t for t in fig.findobj(Text) if t not in ignored and t.get_visible() and t.get_text().strip()]
    outside = []
    for t in texts:
        b=t.get_window_extent(renderer)
        if b.x0 < -1 or b.y0 < -1 or b.x1 > fig.bbox.width+1 or b.y1 > fig.bbox.height+1:
            outside.append(t.get_text())
    if outside: raise RuntimeError(f'{name}: text outside canvas: {outside}')
    overlaps = []
    for i, t in enumerate(texts):
        bt = Text.get_window_extent(t, renderer)
        for u in texts[i+1:]:
            bu = Text.get_window_extent(u, renderer)
            ix = min(bt.x1,bu.x1)-max(bt.x0,bu.x0)
            iy = min(bt.y1,bu.y1)-max(bt.y0,bu.y0)
            if ix > .5 and iy > .5:
                overlaps.append([t.get_text(),u.get_text()])
    if overlaps: raise RuntimeError(f'{name}: text overlap: {overlaps}')
    REPORT['figures'][name]={'text_overlap_pairs':overlaps, 'width_pt':W*72,'height_pt':fig.get_figheight()*72,
        'minimum_font_pt':min(t.get_fontsize() for t in texts), 'text_outside_canvas':outside}
    fig.savefig(PAPER / (name+'.pdf'))
    font_output = subprocess.check_output(['pdffonts', str(PAPER/(name+'.pdf'))], text=True)
    embedded_fonts = [line.split()[0] for line in font_output.splitlines()[2:] if line.strip()]
    if not embedded_fonts or any('TimesNewRoman' not in font for font in embedded_fonts):
        raise RuntimeError(f'{name}: unexpected PDF fonts: {embedded_fonts}')
    REPORT['figures'][name]['embedded_fonts'] = embedded_fonts
    REPORT['figures'][name]['pdf_sha256'] = hashlib.sha256((PAPER/(name+'.pdf')).read_bytes()).hexdigest()
    fig.savefig(PAPER / (name+'.png'), dpi=220)
    plt.close(fig)

def title(ax, text): ax.set_title(text, loc='left', pad=9, fontweight='bold')
def tidy(ax): ax.grid(axis='y',color='#dddddd',lw=.6)
def reference(root):
    for p in sorted(root.glob('run_*/run_meta.json')): read(p)
    i,m=medoid(root)
    REPORT['references'][root.name]={'run':i,'phases':{}}
    for s in m['shifts']:
        for n,r in phases_by_canonical(m,s['shift']).items():
            REPORT['references'][root.name]['phases'][f"shift_{s['shift']}/{n}"]=r['kpis']
    return i,m
SI,SM=reference(SINGLE)
FI,FM=reference(FACTORY)
assert (SI,FI)==(1,8)
S={n:r['kpis'] for n,r in phases_by_canonical(SM,1).items()}
F=read(FACTORY / f'run_{FI:02d}/factory_table.json')

def repeated(root, m, factory=False):
    groups={}
    for p in sorted(root.glob('run_*/run_meta.json')):
        r=read(p)
        if r['status']!='ok': continue
        for s in r['shifts']:
            for q in s['phases']:
                if q.get('canonical') and q.get('schedule') and q.get('kpis'):
                    groups.setdefault((s['shift'],q['phase'],q['canonical']),[]).append(q)
    keys=sorted(groups)
    metrics=[('throughput_uph','Throughput (u/h)'),('fatigue_index','Maximum fatigue'),('noise_db','Noise (dB)')] if factory else [('throughput_uph','Throughput (u/h)'),('vr_score','V/R score'),('fatigue_index','Fatigue index'),('noise_db','Noise (dB)')]
    fig,axes=plt.subplots(3,1,figsize=(W,6.3)) if factory else plt.subplots(2,2,figsize=(W,5.8))
    fig.subplots_adjust(left=.15 if factory else .13,right=.985,top=.945,bottom=.125 if factory else .21,hspace=.8,wspace=.43)
    rng=np.random.default_rng(0)
    for j,(ax,(metric,label)) in enumerate(zip(np.ravel(axes),metrics)):
        vals=[[q['kpis'][metric] for q in groups[k]] for k in keys]
        ax.boxplot(vals,positions=np.arange(len(keys)),widths=.55,showfliers=False,
            patch_artist=True,boxprops={'facecolor':'#edf0f2','edgecolor':'#9ca8ae'},
            medianprops={'color':'#9ca8ae'},whiskerprops={'color':'#9ca8ae'},capprops={'color':'#9ca8ae'})
        for x,k in enumerate(keys):
            q=groups[k]
            sources=[(z.get('pareto') or {}).get('selected_source') for z in q]
            sources=[a if a in ('llm','deterministic') else z['schedule'].get('source','deterministic') for a,z in zip(sources,q)]
            ax.scatter(x+rng.uniform(-.16,.16,len(q)),vals[x],s=15,
                       color=[ORANGE if a=='llm' else BLUE for a in sources],alpha=.8,zorder=3)
            y=phases_by_canonical(m,k[0])[k[2]]['kpis'][metric]
            ax.plot([x-.28,x+.28],[y,y],color='black',lw=1.8,zorder=4)
        ax.set_xticks(range(len(keys)),[k[2] for k in keys], fontsize=9 if factory else 9, rotation=0 if factory else 90)
        ax.set_xlim(-.6,len(keys)-.4)
        ax.set_ylabel(label); title(ax,f'({chr(97+j)}) {label}');tidy(ax)
        if factory:
            lim={'fatigue_index':.4,'noise_db':82}.get(metric)
            if lim:ax.axhline(lim,color='#555555',ls='--',lw=1)
        else:
            for x in [3.5,7.5]:ax.axvline(x,color='#aaaaaa',ls=':',lw=.7)
            for i in range(3):ax.text((1.5+4*i+.6)/12.2,-.29,f'Shift {i+1}',ha='center',va='top',transform=ax.transAxes,fontsize=9)
    handles=[Line2D([],[],marker='o',ls='',color=BLUE,label='Deterministic'),Line2D([],[],marker='o',ls='',color=ORANGE,label='LLM'),Line2D([],[],color='black',lw=2,label='Medoid')]
    if factory:handles.append(Line2D([],[],color='#555555',ls='--',label='Contract limit'))
    fig.legend(handles=handles,loc='lower center',ncol=2 if factory else 3,bbox_to_anchor=(.5,.01),frameon=False)
    save(fig,'fig_repeated_factory' if factory else 'fig_repeated_single')

def learning():
    d=read(SINGLE/f'run_{SI:02d}/multi_shift_learning_curve.json')['shifts']
    fig,axes=plt.subplots(3,1,figsize=(W,5.9));fig.subplots_adjust(left=.16,right=.98,top=.95,bottom=.115,hspace=.85)
    specs=[('vr_score_accepted','(a) Accepted V/R','V/R score'),('total_constraint_violations','(b) Constraints violated by the rejected S2','Count'),('experience_records_after','(c) Cumulative experience','Records')]
    x=[q['shift'] for q in d]
    colors=[GREEN if q['learning_bias_available'] else ORANGE for q in d]
    for ax,(key,t,ylabel) in zip(axes,specs):
        y=[q[key] for q in d]
        if key=='experience_records_after':ax.plot(x,y,'o-',color=BLUE,lw=1.8);ax.fill_between(x,y,color=BLUE,alpha=.12)
        else:ax.bar(x,y,color=colors,width=.55)
        for a,b in zip(x,y):ax.text(a,b+max(y)*.055,f'{b:.3f}' if key=='vr_score_accepted' else str(b),ha='center',va='bottom')
        ax.set_ylim(0,max(y)*1.3);ax.set_xlim(.5,3.5);ax.set_xticks(x,[f'Shift {v}' for v in x]);ax.set_ylabel(ylabel);title(ax,t);tidy(ax)
        if key!='vr_score_accepted':ax.yaxis.set_major_locator(matplotlib.ticker.MaxNLocator(integer=True))
    fig.legend(handles=[Patch(color=ORANGE,label='No prior bias'),Patch(color=GREEN,label='Learning bias active')],ncol=2,loc='lower center',bbox_to_anchor=(.5,.005),frameon=False)
    save(fig,'fig9_learning_curve')

def stochastic():
    d=read(SINGLE/f'run_{SI:02d}/stochastic_verification.json')['schedules'];names=list(d);x=np.arange(len(names))
    fig,(a,b)=plt.subplots(2,1,figsize=(W,5.9));fig.subplots_adjust(left=.15,right=.98,top=.93,bottom=.14,hspace=.78)
    v=[100*d[k]['p_feasible'] for k in names]
    a.bar(x,v,width=.5,color=[GREEN if y>=95 else ORANGE for y in v]);a.axhline(95,color=BLUE,ls='--',lw=1.2)
    for xx,y in zip(x,v):a.text(xx,y+5,f'{y:g}%\n'+('Certified' if y>=95 else 'Rejected'),ha='center',va='bottom')
    a.set_ylim(0,140);a.set_yticks([0,25,50,75,100]);a.set_ylabel('Feasibility probability (%)');title(a,'(a) Monte Carlo feasibility (200 samples)')
    fig.text(.565,.565,'Dashed line: 95% certification threshold',ha='center',va='center',color=BLUE)
    specs=[('FatigueIndex','Fatigue ≤ 0.40','#cf6d65'),('Noise','Noise ≤ 80 dB','#e8a146'),('CyberRiskLevel','Cyber risk ≤ 2','#6caa7a')]
    for j,(key,label,c) in enumerate(specs):
        y=[100*d[k]['violation_probs'].get(key,0) for k in names]
        b.bar(x+(j-1)*.24,y,width=.23,label=label,color=c)
        for xx,yy in zip(x+(j-1)*.24,y):
            if yy:b.text(xx,yy+3,f'{yy:g}%',ha='center',va='bottom',fontsize=9)
    b.set_ylim(0,120);b.set_ylabel('Violation probability (%)');title(b,'(b) Per-constraint violations')
    for ax in (a,b):ax.set_xticks(x,names);tidy(ax)
    b.legend(loc='upper center',bbox_to_anchor=(.5,-.18),ncol=3,frameon=False,fontsize=9,columnspacing=1)
    margins=d['S2']['worst_case_margins'];fig.text(.5,.022,f"S2 fifth-percentile margins: fatigue {margins['FatigueIndex']:+.2f}; noise {margins['Noise']:+.1f} dB",ha='center',fontsize=9)
    save(fig,'fig11_stochastic_verification')

def fatigue():
    fig,ax=plt.subplots(figsize=(W,3.65));fig.subplots_adjust(left=.13,right=.985,top=.84,bottom=.2)
    x=np.arange(4)
    for j,c in enumerate([BLUE,ORANGE,GREEN]):
        y=[F[k][f'fatigue_h{j+1}'] for k in F]
        ax.bar(x+(j-1)*.25,y,width=.23,color=c,label=f'H{j+1}')
        for xx,yy in zip(x+(j-1)*.25,y):ax.text(xx,yy+.016,f'{yy:.3f}',rotation=90,ha='center',va='bottom',fontsize=9,bbox={'facecolor':'white','edgecolor':'none','pad':.7})
    ax.axhline(.4,color='#444444',ls='--',lw=1);ax.axhline(.35,color='#777777',ls=':',lw=1)
    ax.set_ylim(0,.99);ax.set_ylabel('Operator fatigue index');ax.set_xticks(x,list(F));tidy(ax)
    fig.legend(*ax.get_legend_handles_labels(),loc='upper center',ncol=3,frameon=False,bbox_to_anchor=(.5,.985))
    fig.text(.5,.055,'Limits: H1 / H3 = 0.40 (dashed); H2 = 0.35 (dotted)',ha='center')
    save(fig,'fig3_factory_operator_fatigue')

def sensitivity():
    d=read(DATA/'sensitivity/summary.json');fig,axes=plt.subplots(2,2,figsize=(W,5.85))
    fig.subplots_adjust(left=.13,right=.97,top=.92,bottom=.18,wspace=.48,hspace=.67)
    styles=['-','-','--',':','-.'];markers=['o','s','^','D','v']
    for row,case in enumerate(['single','factory']):
        for col,(family,metric,limit) in enumerate([('fatigue_alpha','fatigue',.4),('noise_coeffs','noise',80 if row==0 else 82)]):
            ax=axes[row,col];ends=[]
            for j,n in enumerate(d['cases'][case]['nominal']):
                y=[d['cases'][case]['grid'][family][str(delta)][n][metric] for delta in d['deltas']]
                c=PURPLE if n=='S3d' else (ORANGE if n in ('S2','FS2') else BLUE)
                ax.plot(np.array(d['deltas'])*100,y,ls=styles[j],marker=markers[j],markersize=3,color=c,lw=1.1)
                ends.append((y[-1],n,c))
            ax.axhline(limit,color='#666666',ls='--',lw=.8)
            ax.set_xlim(-33,54);ax.set_xticks([-30,0,30]);ax.set_xlabel('Coefficient error (%)');ax.set_ylabel('Fatigue index' if col==0 else 'Noise (dB)')
            title(ax,f"({chr(97+row*2+col)}) {'Cell' if row==0 else 'Factory'}: {'fatigue' if col==0 else 'noise'}")
            tidy(ax);lo,hi=ax.get_ylim();span=hi-lo;ax.set_ylim(lo,hi+span*.12)
            # Place endpoint labels with a minimum 12 pt vertical gap, then
            # connect displaced labels to their actual endpoints.
            fig.canvas.draw();gap=(ax.get_ylim()[1]-lo)*12/(ax.bbox.height/fig.dpi*72)
            prev=lo
            for val,n,c in sorted(ends):
                y=max(val,prev+gap);prev=y
                ax.annotate(n,xy=(30,val),xytext=(36,y),textcoords='data',ha='left',va='center',color=c,fontsize=9,bbox={'facecolor':'white','edgecolor':'none','pad':.6},
                    arrowprops={'arrowstyle':'-','color':c,'lw':.6},annotation_clip=False)
    fig.legend(handles=[Line2D([],[],color=BLUE,label='Deployed'),Line2D([],[],color=ORANGE,label='Rejected'),Line2D([],[],color=PURPLE,label='Uncertified anchor')],loc='lower center',bbox_to_anchor=(.5,.025),ncol=3,frameon=False,fontsize=9,columnspacing=1)
    save(fig,'fig_sensitivity')

def box(ax,x,y,w,h,text,color='#edf3f7',fontsize=9):
    ax.add_patch(FancyBboxPatch((x,y),w,h,boxstyle='round,pad=0.007',facecolor=color,edgecolor='#637482',lw=.8))
    ax.text(x+w/2,y+h/2,text,ha='center',va='center',fontsize=fontsize,linespacing=1.35)
def arrow(ax,a,b):ax.annotate('',xy=b,xytext=a,arrowprops={'arrowstyle':'->','color':'#4b5b68','lw':1})

def swimlane():
    fig,ax=plt.subplots(figsize=(W,6.0));fig.subplots_adjust(left=.02,right=.98,top=.985,bottom=.02);ax.set(xlim=(0,1),ylim=(0,1));ax.axis('off')
    trace=read(FACTORY/f'run_{FI:02d}/execution_trace.json')
    modes=[q['working_mode'].capitalize() for q in trace]
    notes=['Initial contract: 9 constraints','Demand disturbance: +20%','Replan with H2 absent','Manager authorises contract C2','Adaptation and stabilisation','Shift-close contract revision']
    layers=[q['active_layers'] for q in trace]
    for i,(mode,note,layer) in enumerate(zip(modes,notes,layers)):
        y=.85-i*.163;c={'Legislator':'#e2effa','Auditor':'#e5f1e8','Partner':'#fff0df'}[mode]
        box(ax,.02,y,.22,.125,f'Phase {i+1}\n{mode}',c)
        box(ax,.30,y,.67,.125,note+'\n'+layer,c)
        arrow(ax,(.25,y+.0625),(.29,y+.0625))
        if i<5:arrow(ax,(.635,y-.006),(.635,y-.031))
    save(fig,'fig4_factory_execution_swimlane')

def integrated():
    fig=plt.figure(figsize=(W,6.0))
    fig.text(.035,.979,'(a) Single-cell schedule evaluations',fontweight='bold',fontsize=10,va='top')
    for j,(key,t,lim) in enumerate([('throughput_uph','Throughput (u/h)',None),('noise_db','Noise (dB)',80),('fatigue_index','Fatigue index',.4)]):
        ax=fig.add_axes([.105+j*.305,.77,.23,.145]);v=[S[k][key] for k in ['S1','S2','S3']]
        ax.bar(range(3),v,color=[BLUE,ORANGE,BLUE],width=.55);ax.set_xticks(range(3),['S1','S2','S3']);ax.set_title(t,fontsize=9,pad=7);ax.set_ylim(0,max(v)*1.17);tidy(ax)
        if lim:ax.axhline(lim,color='#555555',ls='--',lw=1)
    fig.text(.5,.719,'S1 / S3: accepted; S2: rejected; dashed lines: contract limits',ha='center',fontsize=9)
    fig.text(.035,.674,'(b) Working-mode lifecycle',fontweight='bold',fontsize=10)
    ax=fig.add_axes([.035,.566,.93,.084]);ax.axis('off');ax.set(xlim=(0,1),ylim=(0,1))
    for i,m in enumerate(['Legislator','Auditor','Auditor','Partner','Auditor','Legislator']):
        c={'Legislator':'#e2effa','Auditor':'#e5f1e8','Partner':'#fff0df'}[m]
        box(ax,i/6+.007,.08,1/6-.018,.78,f'P{i+1}\n{m}',c,9)
    fig.text(.035,.526,'(c) OPC-UA nodes: reference stabilisation values',fontweight='bold',fontsize=10)
    ax=fig.add_axes([.035,.276,.93,.22]);ax.axis('off')
    cell=[('Throughput',f"{S['S3']['throughput_uph']:.1f} u/h"),('Defect rate',f"{S['S3']['defect_rate']:.4f}"),('Noise',f"{S['S3']['noise_db']:.1f} dB"),('Fatigue',f"{S['S3']['fatigue_index']:.3f}"),('Energy',f"{S['S3']['energy_kwh']:.0f} kWh"),('Deadline gap',f"{S['S3']['deadline_gap_pct']:.1f}%"),('Contract','C2 / ACTIVE')]
    factory=[('Cell imbalance',f"{F['FS3']['cell_balance_loss_pct']:.1f}%"),('AGV utilisation',f"{F['FS3']['agv_utilization']:.3f}"),('Fatigue H1',f"{F['FS3']['fatigue_h1']:.3f}"),('Fatigue H2',f"{F['FS3']['fatigue_h2']:.3f}"),('Fatigue H3',f"{F['FS3']['fatigue_h3']:.3f}"),('Noise',f"{F['FS3']['noise_db']:.1f} dB"),('Cyber risk','Medium')]
    for x,head,rows in [(0,'Single cell / S3',cell),(.53,'Factory / FS3',factory)]:
        ax.text(x,.98,head,va='top',fontweight='bold',fontsize=9)
        for i,(k,v) in enumerate(rows):
            y=.79-i*.12;ax.text(x,y,k,va='center',fontsize=9);ax.text(x+.45,y,v,va='center',ha='right',fontsize=9)
    fig.text(.035,.235,'(d) Execution-stack data flow',fontweight='bold',fontsize=10)
    ax=fig.add_axes([.035,.015,.93,.19]);ax.axis('off');ax.set(xlim=(0,1),ylim=(0,1))
    box(ax,.01,.64,.24,.3,'CBPA\nL1–L5 + Meta');box(ax,.38,.64,.24,.3,'Isaac Sim\ndigital twin');box(ax,.75,.64,.24,.3,'BaSyx AAS\nregistry')
    ax.annotate('',xy=(.37,.79),xytext=(.26,.79),arrowprops={'arrowstyle':'<->','color':'#4b5b68','lw':1});ax.annotate('',xy=(.74,.79),xytext=(.63,.79),arrowprops={'arrowstyle':'<->','color':'#4b5b68','lw':1})
    ax.text(.315,.97,'Deploy / read',ha='center',va='bottom',fontsize=9);ax.text(.685,.97,'AAS sync',ha='center',va='bottom',fontsize=9)
    box(ax,.01,.20,.24,.26,'OPC-UA\nserver');box(ax,.38,.20,.24,.26,'PLC / SCADA\nclient');box(ax,.75,.20,.24,.26,'Streamlit\ndashboard')
    arrow(ax,(.13,.63),(.13,.47));ax.text(.16,.55,'Publish',va='center',fontsize=9)
    arrow(ax,(.26,.33),(.37,.33));ax.text(.315,.51,'Subscribe',ha='center',fontsize=9)
    ax.plot([.13,.13,.87],[.19,.075,.075],color='#4b5b68',lw=1);arrow(ax,(.87,.075),(.87,.19));ax.text(.5,.025,'Read',ha='center',va='center',fontsize=9)
    save(fig,'fig_integrated_stack')

if __name__=='__main__':
    repeated(FACTORY,FM,True);repeated(SINGLE,SM);learning();stochastic();fatigue();sensitivity();swimlane();integrated()
    for path,h in REPORT['inputs'].items():
        assert hashlib.sha256((CASE/path).read_bytes()).hexdigest()==h
    REPORT['renderer_sha256'] = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
    REPORT['medoid_helper_sha256'] = hashlib.sha256((CASE/'scripts/paper_values.py').read_bytes()).hexdigest()
    (PAPER/'figure_render_audit.json').write_text(json.dumps(REPORT,indent=2)+'\n')
    print(json.dumps({'figures':REPORT['figures'],'input_files_unchanged':len(REPORT['inputs'])},indent=2))
