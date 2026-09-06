from pathlib import Path
from collections import Counter,defaultdict
from datetime import datetime
import matplotlib.pyplot as plt
from matplotlib import font_manager
from .analytics import stats,theme_counts,recent_negative_alert

def apply_font(config):
    candidates=config.get('visualization',{}).get('font_candidates',[])
    available={f.name for f in font_manager.fontManager.ttflist}
    selected=next((x for x in candidates if x in available),'DejaVu Sans')
    plt.rcParams['font.family']=selected; plt.rcParams['axes.unicode_minus']=False
    return selected

def make_charts(rows,out_dir,config):
    apply_font(config); out=Path(out_dir); out.mkdir(parents=True,exist_ok=True)
    # 1 sentiment
    c=Counter(r.get('sentiment') or 'unanalyzed' for r in rows)
    labels=['positive','neutral','negative','unanalyzed']; vals=[c[x] for x in labels]
    plt.figure(figsize=(8,5)); plt.bar(labels,vals); plt.title('Sentiment Distribution'); plt.xlabel('Sentiment'); plt.ylabel('Reviews'); plt.tight_layout()
    p1=out/'sentiment_distribution.png'; plt.savefig(p1,dpi=150); plt.close()
    # 2 monthly trend
    monthly=defaultdict(Counter)
    for r in rows:
        d=r.get('review_date'); s=r.get('sentiment')
        if d and s: monthly[d[:7]][s]+=1
    months=sorted(monthly)
    plt.figure(figsize=(12,5))
    for s in ['positive','neutral','negative']:
        plt.plot(months,[monthly[m][s] for m in months],label=s)
    plt.title('Monthly Sentiment Trend'); plt.xlabel('Month'); plt.ylabel('Reviews'); plt.xticks(rotation=60); plt.legend(); plt.tight_layout()
    p2=out/'sentiment_trend.png'; plt.savefig(p2,dpi=150); plt.close()
    # 3 rating sentiment matrix
    ratings=[1,2,3,4,5]; sentiments=['positive','neutral','negative']
    matrix=[[sum(1 for r in rows if r.get('rating') is not None and int(round(float(r['rating'])))==ra and r.get('sentiment')==s) for s in sentiments] for ra in ratings]
    import numpy as np
    arr=np.array(matrix)
    plt.figure(figsize=(8,5)); plt.imshow(arr,aspect='auto'); plt.title('Rating vs Sentiment Matrix'); plt.xlabel('Sentiment'); plt.ylabel('Rating'); plt.xticks(range(3),sentiments); plt.yticks(range(5),ratings)
    for i in range(5):
        for j in range(3): plt.text(j,i,str(arr[i,j]),ha='center',va='center')
    plt.tight_layout(); p3=out/'rating_sentiment_matrix.png'; plt.savefig(p3,dpi=150); plt.close()
    return [str(p1),str(p2),str(p3)]

def build_report(rows,latest_extract=None,top_n=5,config=None):
    st=stats(rows); neg=theme_counts(rows,'negative').most_common(top_n); pos=theme_counts(rows,'positive').most_common(top_n)
    products=Counter((r.get('product') or 'Unknown') for r in rows).most_common(top_n)
    providers=Counter(r.get('analysis_provider') or 'unanalyzed' for r in rows)
    lines=['# 전자제품 고객 리뷰 감정 분석 대시보드','',f"생성일시: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",'', '## 핵심 지표',
      f"- 총 리뷰 수: {st['total']:,}건",f"- 분석 완료율: {st['analysis_rate']:.1f}%",f"- 긍정 비율: {st['positive_ratio']:.1f}%",f"- 부정 비율: {st['negative_ratio']:.1f}%",
      f"- 평균 별점: {st['average_rating']:.2f}" if st['average_rating'] is not None else '- 평균 별점: N/A',
      f"- 평균 감정 신뢰도: {st['average_confidence']:.2f}" if st['average_confidence'] is not None else '- 평균 감정 신뢰도: N/A',
      f"- 별점-감정 일치율: {st['rating_sentiment_agreement']:.1f}%" if st['rating_sentiment_agreement'] is not None else '- 별점-감정 일치율: N/A',
      f"- 저별점(1~2점) 비율: {st['low_rating_ratio']:.1f}%", f"- 분석 방식: {dict(providers)}",'',
      '## TOP 긍정 테마']
    lines += [f"{i}. {k}: {v}건" for i,(k,v) in enumerate(pos,1)] or ['- 없음']
    lines += ['', '## TOP 부정 테마']+[f"{i}. {k}: {v}건" for i,(k,v) in enumerate(neg,1)] or ['- 없음']
    lines += ['',f'## TOP {top_n} 리뷰 제품']+[f"{i}. {name}: {cnt}건" for i,(name,cnt) in enumerate(products,1)]
    if latest_extract:
        lines += ['', '## AI/추출 인사이트',f"- 제공자: {latest_extract.get('provider')}",
          '- 긍정 키워드: '+', '.join(latest_extract.get('positive_keywords',[])),
          '- 부정 키워드: '+', '.join(latest_extract.get('negative_keywords',[])),
          '- 요약: '+latest_extract.get('summary',''),'- 개선 제안: '+latest_extract.get('suggestions','')]
    alert=recent_negative_alert(rows,config.get('alert',{}).get('recent_days',30),config.get('alert',{}).get('negative_ratio_threshold',0.30)) if config else None
    if alert:
        lines += ['', '## 최근 부정 리뷰 경고', f"- 기간: {alert['start']} ~ {alert['end']}",f"- 리뷰 수: {alert['count']}건",f"- 부정 비율: {alert['negative_ratio']*100:.1f}%", f"- 경고: {'YES' if alert['warning'] else 'NO'}"]
    return '\n'.join(lines)

def dashboard(storage,config,filters=None,out_dir=None,top_n=5,html=False):
    rows=storage.filtered(filters); out=Path(out_dir or config.get('visualization',{}).get('output_dir','output')); out.mkdir(parents=True,exist_ok=True)
    charts=make_charts(rows,out,config); report=build_report(rows,storage.latest_extract(filters),top_n,config)
    (out/'report.md').write_text(report,encoding='utf-8'); (out/'report.txt').write_text(report.replace('#',''),encoding='utf-8')
    html_path=None
    if html:
        body=report.replace('&','&amp;').replace('<','&lt;').replace('>','&gt;').replace('\n','<br>')
        imgs=''.join(f'<h3>{Path(p).name}</h3><img src="{Path(p).name}" style="max-width:900px;width:100%">' for p in charts)
        content=f'<!doctype html><meta charset="utf-8"><title>Review Dashboard</title><body style="font-family:sans-serif;max-width:1000px;margin:auto"><h1>전자제품 고객 리뷰 대시보드</h1><div>{body}</div>{imgs}</body>'
        html_path=out/'dashboard.html'; html_path.write_text(content,encoding='utf-8')
    return {'charts':charts,'report':str(out/'report.md'),'txt':str(out/'report.txt'),'html':str(html_path) if html_path else None,'text':report}
