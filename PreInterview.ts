function mod(n, m) {
  return ((n % m) + m) % m;
}

function get_weekday(inp){
    var week_day = {0:'Sun',1:'Mon',2:'Tue',3:'Wed',4:'Thu',5:'Fri',6:'Sat'}
    var info = inp.split('-');
    var k = Number(info[2]);
    var c = Number(info[0].substr(0,2))
    if (Number(info[1]) == 1 || Number(info[1]) == 2){
        var m = Number(info[1]) + 10;
        var y = Number(info[0].substr(2,2)) - 1;
    }
    else {
        var m = Number(info[1]) - 2;
        var y = Number(info[0].substr(2,2));
    }
    var w = mod(k + Math.floor(2.6*m-0.2)-2*c+y+Math.floor(y/4)+Math.floor(c/4),7);
    return week_day[w];
}

function solution(D){
    var res = {'Mon':0,'Tue':1,'Wed':2,'Thu':3,'Fri':4,'Sat':5,'Sun':6}
    var val = [null,null,null,null,null,null,null];
    for (let key in D) {
        var day = get_weekday(key);
        let value = D[key];
        val[res[day]] += value;
    }
    for( let i = 0; i <7 ; i++){
        if(val[i] != null){
            continue;
        }else{
            if(val[i] == null && val[i+1] != null){
                val[i] = (val[i-1] + val[i+1])/2;
            }
            else if( val[i]== null && val[i+1]== null && val[i+2] != null){
                val[i] = (val[i-1]*2 + val[i+2])/3; 
                val[i+1] = (val[i-1] + val[i+2]*2)/3;
            }
            else if(val[i]== null && val[i+1]== null && val[i+2] == null && val[i+3] != null){
                val[i] = (val[i-1]*3 + val[i+3])/4; 
                val[i+1] = (val[i-1] + val[i+3])/2;
                val[i+2] = (val[i-1] + val[i+3]*3)/4 ;
            }
            else if(val[i]== null && val[i+1]== null && val[i+2] == null && val[i+3] == null && val[i+4]!= null){
                val[i] = (val[i-1]*4 + val[i+4])/5; 
                val[i+1] = (val[i-1]*3 + val[i+4]*2)/5;
                val[i+2] = (val[i-1]*2 + val[i+4]*3)/5 ;
                val[i] = (val[i-1] + val[i+4]*4)/5;
            }
            else if(val[i]== null && val[i+1]== null && val[i+2] == null && val[i+3] == null && val[i+4] == null && val[i+5]!= null){
                val[i] = (val[i-1]*5 + val[i+5])/6; 
                val[i+1] = (val[i-1]*2 + val[i+5])/3;
                val[i+2] = (val[i-1] + val[i+5])/2 ;
                val[i] = (val[i-1] + val[i+5]*2)/3;
                val[i] = (val[i-1] + val[i+5]*5)/6;
            }
        }
    }
    let count = 0;
    for(let key in res){
        res[key] = val[count];
        count +=1;
    }
    return res;
}

var d = {'2020-01-01':4, '2020-01-02':4, '2020-01-03':6, '2020-01-04':8, '2020-01-05':2,'2020-01-06':-6, '2020-01-07':2, '2020-01-08':-2};
var df = {'2020-01-05':14,'2020-01-06':2};
console.log(solution(d));

