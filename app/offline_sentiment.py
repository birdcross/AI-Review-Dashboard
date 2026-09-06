import math
import re

POS={
'good':1,'great':1.3,'excellent':1.7,'awesome':1.6,'amazing':1.6,'love':1.8,'loved':1.8,'like':0.7,
'nice':0.8,'perfect':1.7,'best':1.5,'better':0.8,'easy':0.9,'comfortable':1,'clear':0.8,'happy':1.2,
'satisfied':1.3,'recommend':1.1,'recommended':1.1,'worth':0.8,'fast':0.6,'solid':0.6,'beautiful':0.9,
'reliable':1.0,'convenient':0.8,'favorite':1.2,'works':0.5,'well':0.4,'smooth':0.7,'responsive':0.8,
'fantastic':1.6,'wonderful':1.6,'impressed':1.2,'pleased':1.1,'affordable':0.7}
NEG={
'bad':-1.4,'poor':-1.3,'terrible':-1.8,'awful':-1.8,'disappointed':-1.5,'disappointing':-1.5,
'hard':-0.7,'difficult':-0.9,'problem':-1.1,'problems':-1.1,'issue':-0.9,'issues':-0.9,
'expensive':-0.6,'worse':-1.2,'worst':-1.8,'broken':-1.6,'defective':-1.7,'slow':-0.7,
'frustrating':-1.4,'annoying':-1.2,'bug':-1.0,'bugs':-1.0,'disconnect':-1.4,'disconnecting':-1.4,
'drops':-1.0,'return':-0.8,'returned':-0.9,'fail':-1.3,'failed':-1.3,'failure':-1.4,
'uncomfortable':-1.1,'weak':-0.8,'stuck':-1.0,'jumpy':-0.8,'dead':-1.5,'hate':-1.7,
'useless':-1.7,'waste':-1.6,'overpriced':-1.1,'flimsy':-0.9,'crash':-1.2}
NEGATORS={'not','no','never','dont',"don't",'doesnt',"doesn't",'didnt',"didn't",'cannot',"can't",'hardly'}
POS_PHRASES={'works great':1.5,'works well':1.1,'easy to use':1.2,'easy to set up':1.2,'easy to install':1.1,
'very good':1.3,'very easy':1.0,'highly recommend':1.8,'worth the money':1.2,'happy with':1.2,
'love this':1.7,'love it':1.7,'no complaints':1.2}
NEG_PHRASES={'does not work':-2.0,'did not work':-2.0,'stopped working':-2.0,'not worth':-1.4,
'too expensive':-1.1,'waste of money':-2.0,'not good':-1.4,'very disappointed':-1.8,
"doesn't work":-2.0,'do not recommend':-1.8,'would not recommend':-1.8,'keeps disconnecting':-1.7}

def analyze(text):
    low=(text or '').lower(); score=0.0; hits=0
    for p,w in POS_PHRASES.items(): c=low.count(p); score+=w*c; hits+=c
    for p,w in NEG_PHRASES.items(): c=low.count(p); score+=w*c; hits+=c
    toks=re.findall(r"[a-z']+",low)
    for i,t in enumerate(toks):
        if t in POS:
            w=POS[t]
            if any(x in NEGATORS for x in toks[max(0,i-3):i]): w=-w
            score+=w; hits+=1
        elif t in NEG:
            w=NEG[t]
            if any(x in NEGATORS for x in toks[max(0,i-3):i]): w=-w*0.7
            score+=w; hits+=1
    score=score/max(1.0,math.sqrt(max(1,len(toks))/12))
    sentiment='positive' if score>=0.55 else 'negative' if score<=-0.55 else 'neutral'
    confidence=min(0.97,max(0.55,0.58+min(abs(score),3)*0.11+min(hits,5)*0.025))
    return sentiment,round(confidence,3)
