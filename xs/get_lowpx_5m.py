import pickle, yfinance as yf, warnings; warnings.filterwarnings('ignore')
SC='/tmp/claude-0/-home-claude/6d3c83e1-1a4a-5a0a-8647-c2590d1d30a4/scratchpad/'; D=pickle.load(open(SC+'lowpx_daily.pkl','rb')); T=sorted(D); out={}
for i in range(0,len(T),100):
    try:
        d=yf.download(T[i:i+100],period='60d',interval='5m',prepost=True,group_by='ticker',auto_adjust=False,progress=False,threads=True)
        for t in T[i:i+100]:
            try:
                x=d[t].dropna(subset=['Close'])
                if len(x)>500: out[t]=x
            except Exception: pass
    except Exception as e: print('err',i,e)
    print(i,len(out),flush=True)
pickle.dump(out,open(SC+'lowpx_5m.pkl','wb')); print('saved',len(out))
