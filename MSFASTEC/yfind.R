
y.find <- function(b){
  Y.fcst.96 <- c()
  train =(1):(200*96)
  test = (200*96+1):(365*96)
  wdw.seq.1 <- seq(1,(200*96),by=96) 
  wdw.seq.2 <- seq(96,(200*96+1),by=96)
  for(i in 2:200){
    wdw.1 <- (1+((i-30)-1)*96):((i-1)*96) 
    wdw.2 <- ((i-1)*96+1-9):((i)*96-9)
    mu.hat <-  exp.a
    ex.1 <- XX[wdw.2,1]
    ex.2 <- XX[wdw.2,2]
    ex.3 <- XX[wdw.2,3]
    ex.4 <- y.trash[wdw.2+9]
    j   <- ((i-1)*96)
    k.1 <- j+1
    k.2 <- j+96
    t.stat <- which(((abs(mean(Y[wdw.2])-mu.hat))/min(sqrt((mean(Y[wdw.2])-mu.hat)^2))-1)==0)
    Y.fcst.96[wdw.seq.1[i]:wdw.seq.2[i]] <- if(t.stat==1){
      b[1]*mu.hat[t.stat]+b[17]*ex.1+b[18]*ex.2+b[19]*ex.3+b[20]*ex.4
    }else if(t.stat==2){
      b[2]*fcst.curve.DK1[[7]][1:96,1]+b[17]*ex.1+b[18]*ex.2+b[19]*ex.3+b[20]*ex.4
    }else if(t.stat==3){
      b[3]*fcst.curve.DK2[[7]][1:96,1]+b[17]*ex.1+b[18]*ex.2+b[19]*ex.3+b[20]*ex.4
    }else if(t.stat==4){
      b[4]*fcst.curve.DK3[[7]][1:96,1]+b[17]*ex.1+b[18]*ex.2+b[19]*ex.3+b[20]*ex.4
    }else if(t.stat==5){
      b[5]*mu.hat[t.stat]+b[17]*ex.1+b[18]*ex.2+b[19]*ex.3+b[20]*ex.4
    }else if(t.stat==6){
      b[6]*mu.hat[t.stat]+b[17]*ex.1+b[18]*ex.2+b[19]*ex.3+b[20]*ex.4
    }else if(t.stat==7){
      b[7]*fcst.curve.DK4[[7]][1:96,1]+b[17]*ex.1+b[18]*ex.2+b[19]*ex.3+b[20]*ex.4
    }else if(t.stat==8){
      b[8]*mu.hat[t.stat]+b[17]*ex.1+b[18]*ex.2+b[19]*ex.3+b[20]*ex.4
    }else if(t.stat==9){
      b[9]*fcst.curve.DK5[[7]][1:96,1]+b[17]*ex.1+b[18]*ex.2+b[19]*ex.3+b[20]*ex.4
    }else if(t.stat==10){
      b[10]* fcst.curve.DK6[[7]][1:96,1]+b[17]*ex.1+b[18]*ex.2+b[19]*ex.3+b[20]*ex.4
    }else if(t.stat==11){
      b[11]*fcst.curve.DK7[[7]][1:96,1]+b[17]*ex.1+b[18]*ex.2+b[19]*ex.3+b[20]*ex.4
    }else if(t.stat==12){
      b[12]* fcst.curve.DK8[[7]][1:96,1]+b[17]*ex.1+b[18]*ex.2+b[19]*ex.3+b[20]*ex.4
    }else if(t.stat==13){
      b[13]* fcst.curve.DK9[[7]][1:96,1]+b[17]*ex.1+b[18]*ex.2+b[19]*ex.3+b[20]*ex.4
    }else if(t.stat==14){
      b[14]* fcst.curve.DK10[[7]][1:96,1]+b[17]*ex.1+b[18]*ex.2+b[19]*ex.3+b[20]*ex.4
    }else if(t.stat==15){
      b[15]* mu.hat[t.stat]+b[17]*ex.1+b[18]*ex.2+b[19]*ex.3+b[20]*ex.4
    }else if(t.stat==16){
      b[16]* mu.hat[t.stat]+b[17]*ex.1+b[18]*ex.2+b[19]*ex.3+b[20]*ex.4
    }
  }
  Y.fcst.96 <- Y.fcst.96[-c(1:(96))]
  value <- mean(abs((1+exp(-Y.train[(1*96+1):(length(Y.train))]))^(-1)-(1+exp(-Y.fcst.96))^(-1)))
  return(value)
}