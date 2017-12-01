%% Task 3 solution
% Adam Morris ? Computational Social Cognition Bootcamp, July 2017

priorGuilty = .9; % prior probability that he's guilty
probObsGivenGuilty = .9; % if he's guilty, what's the probability that any one person would have seen him there
probObsGivenInnocent = .2; % if he's NOT guilty, what's the probability that any one person would have seen him there

% Returns prob(guilty, numObs | numPeopleInPark) - the numerator in Bayes
% rule
% numObs corresponds to 'R' in the task, and numPeopleInPark corresponds to
% N
getNumerator = @(priorOfHypothesis, numObs, numPeopleInPark, probsObsGivenHypothesis) ...
    priorOfHypothesis * binopdf(numObs, numPeopleInPark, probsObsGivenHypothesis);

% Returns the posterior probability that he's guilty,
% given a fixed number of sightings AND a fixed number of people in the
% park
% i.e. prob(guilty | numObs, numPeopleInPark)
getPosterior = @(priorGuilty, numObs, numPeopleInPark) ...
    getNumerator(priorGuilty, numObs, numPeopleInPark, probObsGivenGuilty) ./ ... % numerator of Bayes rule
    (getNumerator(priorGuilty, numObs, numPeopleInPark, probObsGivenGuilty) + getNumerator(1 - priorGuilty, numObs, numPeopleInPark, probObsGivenInnocent)); % denominator

%% Part A
% Find the posterior for different numbers of sightings (given N = 7)
posteriors = zeros(8, 1);

for numObs = 0:(size(posteriors, 1) - 1)
    posteriors(numObs + 1, 1) = getPosterior(priorGuilty, numObs, 7);
end

% Plot results.
figure
plot(0:(size(posteriors, 1) - 1), posteriors, 'LineWidth', 4);
xlabel('Number of observations');
ylabel('Probability of guilt \newline (assuming N = 7)');
set(gca, 'LineWidth', 2);
set(gca, 'FontSize', 36);

% Probability that he's guilty, given NO observations (and N = 7)
posteriors(1)

%% Part B
% Compute the MARGINAL probability of guilt, given fixed number of sightings:
% p(guilty | zero sightings)

rangePeopleInPark = 0:1000;
getMarginal = @(priorGuilty, numObs, avgPeopleInPark) ...
    sum(getPosterior(priorGuilty, numObs, rangePeopleInPark) .* ...
        poisspdf(rangePeopleInPark, avgPeopleInPark));
    
% What is the marginal probability given zero sightings?
numObs = 0;
avgPeopleInPark = 7;
getMarginal(priorGuilty, numObs, avgPeopleInPark)    
   
%% Part C
% What is the marginal probability given zero sightings, if the average
% people in the park is 1?
getMarginal(priorGuilty, numObs, 1)

% Plot for a bunch of different average people in park.
marginals = zeros(7, 1);
for avgPeopleInPark = 1:size(marginals, 1)
    marginals(avgPeopleInPark) = getMarginal(priorGuilty, numObs, avgPeopleInPark);
end

figure
plot(1:size(marginals, 1), marginals, 'LineWidth', 4);
xlabel('Average people in park');
ylabel('Marginal probability of guilt');
set(gca, 'LineWidth', 2);
set(gca, 'FontSize', 36);

%% Part D
probObsGivenGuilty = .4; % what if he's a master of disguise?

% Recompute joint / posterior / marginal
getNumerator = @(priorOfHypothesis, numObs, numPeopleInPark, probsObsGivenHypothesis) ...
    priorOfHypothesis * binopdf(numObs, numPeopleInPark, probsObsGivenHypothesis);

getPosterior = @(priorGuilty, numObs, numPeopleInPark) ...
    getNumerator(priorGuilty, numObs, numPeopleInPark, probObsGivenGuilty) ./ ... % numerator of Bayes rule
    (getNumerator(priorGuilty, numObs, numPeopleInPark, probObsGivenGuilty) + getNumerator(1 - priorGuilty, numObs, numPeopleInPark, probObsGivenInnocent)); % denominator

getMarginal = @(priorGuilty, numObs, avgPeopleInPark) ...
    sum(getPosterior(priorGuilty, numObs, rangePeopleInPark) .* ...
        poisspdf(rangePeopleInPark, avgPeopleInPark));
    
% Now what is the marginal probability given zero sightings, if the average
% people in the park is 1?
getMarginal(priorGuilty, numObs, 1)