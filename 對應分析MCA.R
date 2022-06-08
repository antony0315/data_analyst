install.packages("ca")
library(ca)
data("smoke")
(result<-chisq.test(smoke))


par(family="STKaiti")
mosaicplot(smoke,main="",color=c("red","blue","green","orange"))
smca<-ca(smoke)
summary(smca)
plot(smca,main="smoke data")



library(FactoMineR)
library(factoextra)
data("UCBAdmissions")
mca<-mjca(UCBAdmissions)
summary(mca)
par(family="STKaiti")
plot(mca,mass=c(TRUE,TRUE),col=c("black","red","green","blue"),
     main="??????????????????")




