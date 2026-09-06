from collections import Counter
from datetime import datetime, timedelta
import re

STOP=set("the and for with this that was are but not have from you your they them its our my i we he she to of in on is it a an as at be been being or if so very just one all can could would should will has had do does did about than then when where what which who how up out over into after before more most some any each other only also really much too very product use used using get got make made bought buy new time because there were still want need".split())

def stats(rows):
    total=len(rows); analyzed=[r for r in rows if r.get('sentiment')]
    sentiments=Counter(r.get('sentiment') for r in analyzed)
    ratings=[float(r['rating']) for r in rows if r.get('rating') is not None]
    conf=[float(r['confidence']) for r in analyzed if r.get('confidence') is not None]
    rating_sent=[]
    agree=0
    for r in analyzed:
        if r.get('rating') is None: continue
        rt='positive' if float(r['rating'])>=4 else 'negative' if float(r['rating'])<=2 else 'neutral'
        rating_sent.append(rt)
        agree += int(rt==r['sentiment'])
    return {
        'total':total,'analyzed':len(analyzed),'analysis_rate':len(analyzed)/total*100 if total else 0,
        'positive':sentiments['positive'],'neutral':sentiments['neutral'],'negative':sentiments['negative'],
        'positive_ratio':sentiments['positive']/len(analyzed)*100 if analyzed else 0,
        'neutral_ratio':sentiments['neutral']/len(analyzed)*100 if analyzed else 0,
        'negative_ratio':sentiments['negative']/len(analyzed)*100 if analyzed else 0,
        'rating_counts':Counter(int(round(x)) for x in ratings),'average_rating':sum(ratings)/len(ratings) if ratings else None,
        'average_confidence':sum(conf)/len(conf) if conf else None,
        'rating_sentiment_agreement':agree/len(rating_sent)*100 if rating_sent else None,
        'low_rating_ratio':sum(x<=2 for x in ratings)/len(ratings)*100 if ratings else 0,
    }

def top_words(rows,sentiment=None,n=8):
    cnt=Counter()
    for r in rows:
        if sentiment and r.get('sentiment')!=sentiment: continue
        text=r.get('original_text') or r.get('review_text') or ''
        words=[w for w in re.findall(r"[a-z][a-z'-]+",text.lower()) if len(w)>2 and w not in STOP]
        cnt.update(words)
    return cnt.most_common(n)

def theme_counts(rows,sentiment=None):
    themes={
      '연결/블루투스':['bluetooth','connection','connect','disconnect','pairing','wifi','wi-fi'],
      '배터리':['battery','charge','charging'],
      '음질/사운드':['sound','audio','bass','speaker','volume'],
      '화면/터치':['screen','display','touch','picture'],
      '버튼/리모컨':['remote','button','buttons'],
      '설정/설치':['setup','set up','install','program'],
      '가격':['price','expensive','money','value'],
      '품질/고장':['quality','broken','defective','fail','failed','problem','issue'],
      '착용감':['comfortable','fit','headphones','earbuds'],
    }
    counts=Counter()
    for r in rows:
        if sentiment and r.get('sentiment')!=sentiment: continue
        text=(r.get('original_text') or r.get('review_text') or '').lower()
        for theme,keys in themes.items():
            if any(k in text for k in keys): counts[theme]+=1
    return counts

def recent_negative_alert(rows,recent_days=30,threshold=0.30):
    dates=[r.get('review_date') for r in rows if r.get('review_date')]
    if not dates:return None
    max_date=max(dates)
    try:end=datetime.strptime(max_date,'%Y-%m-%d')
    except:return None
    start=end-timedelta(days=recent_days-1)
    recent=[r for r in rows if r.get('review_date') and start.strftime('%Y-%m-%d')<=r['review_date']<=max_date and r.get('sentiment')]
    if not recent:return None
    ratio=sum(r['sentiment']=='negative' for r in recent)/len(recent)
    return {'start':start.strftime('%Y-%m-%d'),'end':max_date,'count':len(recent),'negative_ratio':ratio,'warning':ratio>=threshold}
