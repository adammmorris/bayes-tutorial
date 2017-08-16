%% Task 2 solution
% Adam Morris ? Computational Social Cognition Bootcamp, July 2017

%% Part A
% Compute numerator
getJoint = @(prior, gameResults, competence) ...
    prior * binopdf(sum(gameResults), length(gameResults), competence);

%% Part B
% Do Riemann approximation to denominator, get posterior
stepSize = .01;
steps = 0 : stepSize : 1;
getPosterior = @(prior, gameResults, competence) ...
    getJoint(prior, gameResults, competence) / ...
    sum(getJoint(prior, gameResults, steps) * stepSize);

competenceRange = 0: .01 : 1;
plot(competenceRange, getPosterior(1, [0 0 0 0 0 0 1], competenceRange), 'LineWidth', 4);
xlabel('Competence');
ylabel('Posterior over competence');
set(gca, 'LineWidth', 2);
set(gca, 'FontSize', 36);

%% Part C
% Implement pool sharks
sharkCutoff = .9;
getProbWinFromCompetence = @(competence) competence .* (competence < sharkCutoff) + .2 * (competence >= sharkCutoff);
% Plot new relationship between competence and win probability
plot(competenceRange, getProbWinFromCompetence(competenceRange), 'LineWidth', 4);
xlabel('Skill');
ylabel('Probability of winning');
set(gca, 'LineWidth', 2);
set(gca, 'FontSize', 36);

% Redo posterior
getJoint = @(prior, gameResults, competence) ...
    prior * binopdf(sum(gameResults), length(gameResults), getProbWinFromCompetence(competence));
getPosterior = @(prior, gameResults, competence) ...
    getJoint(prior, gameResults, competence) / ...
    sum(getJoint(prior, gameResults, steps) * stepSize);

% Plot posterior
plot(competenceRange, getPosterior(1, [0 0 0 0 0 0 0 0 0 1], competenceRange), 'LineWidth', 4);
xlabel('Competence');
ylabel('Posterior over competence');
set(gca, 'LineWidth', 2);
set(gca, 'FontSize', 36);

%% Part D
% Implement beginner's luck
luckCutoff = .2;
getProbWinFromCompetence = @(competence) ...
    competence .* (competence < sharkCutoff & competence > luckCutoff) + .2 * (competence >= sharkCutoff) + ...
    .5 * (competence <= luckCutoff);
plot(competenceRange, getProbWinFromCompetence(competenceRange));

getJoint = @(prior, gameResults, competence) ...
    prior * binopdf(sum(gameResults), length(gameResults), getProbWinFromCompetence(competence));
getPosterior = @(prior, gameResults, competence) ...
    getJoint(prior, gameResults, competence) / ...
    sum(getJoint(prior, gameResults, steps) * stepSize);

plot(competenceRange, getPosterior(1, [0 0 0 0 0 0 0 0 0 1], competenceRange), 'LineWidth', 4);
xlabel('Competence');
ylabel('Posterior over competence');
set(gca, 'LineWidth', 2);
set(gca, 'FontSize', 36);