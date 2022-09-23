%% clear workspace, remove figures and load data
clc
clear
close all

%load data
weather = load('weather.txt');

%70% of all the obersvation goes to trainingSet (10 in case of 14 observation)
trainingSetRowsNumber = round(0.7*size(weather,1));

%select a values (0 means that there is no laplace smoothing)
a = [0;1;0.5;2];


classification=zeros(size(weather,1)-trainingSetRowsNumber,10);
errorRate=zeros(size(a,1),10);

for i=1:10
    
    % select 10 random rows for trainingSet
    randomRows=randperm(size(weather,1),trainingSetRowsNumber);
    trainingSet=weather(randomRows,:);
    
    % select the remaining rows for dataSet
    dataSet = weather(setdiff(1:end,randomRows),:);
    
    %execute with the four a values
    for k=1:4
        [classification(:,i),errorRate(k,i)]=myBayesClassifier(trainingSet,dataSet,a(k));
        
    end
    
end

%% plot results

bar(errorRate');
legend('Without Laplace Smoothing','Laplace Smoothing a = 1','Laplace Smoothing a = 0.5','Laplace Smoothing a = 2');
xlabel('random test subset');
ylabel('Error rate');
title('Error rate with and without Laplace Smoothing');
set(gca, 'YGrid', 'on', 'XGrid', 'off');