document.addEventListener('DOMContentLoaded', () => {
    fetch('/dropdown_data')
        .then(response => response.json())
        .then(data => {
            const team1Dropdown = document.getElementById('team1');
            data.team1.forEach(team => {
                const option = document.createElement('option');
                option.value = team.name;
                option.textContent = team.name;
                team1Dropdown.appendChild(option);
            });

            const team2Dropdown = document.getElementById('team2');
            data.team2.forEach(team => {
                const option = document.createElement('option');
                option.value = team.name;
                option.textContent = team.name;
                team2Dropdown.appendChild(option);
            });

            const cityDropdown = document.getElementById('city');
            data.cities.forEach(city => {
                const option = document.createElement('option');
                option.value = city;
                option.textContent = city;
                cityDropdown.appendChild(option);
            });
        })
        .catch(error => {
            console.error("Error loading data:", error);
            showAlert("Error loading dropdown data. Please try again later.");
        });
});

function showAlert(message) {
    const errorMessage = document.getElementById('error-message');
    errorMessage.textContent = message;
    errorMessage.style.display = 'block';

    setTimeout(() => {
        errorMessage.style.display = 'none';
    }, 3000);
}

function updateTossOptions() {
    const team1 = document.getElementById('team1').value;
    const team2 = document.getElementById('team2').value;
    const tossWinnerSelect = document.getElementById('toss-winner');
    tossWinnerSelect.innerHTML = `<option value="" disabled selected>Select Toss Winner</option>`;

    if (team1 && team2) {
        const option1 = document.createElement('option');
        option1.value = team1;
        option1.textContent = team1;

        const option2 = document.createElement('option');
        option2.value = team2;
        option2.textContent = team2;

        tossWinnerSelect.appendChild(option1);
        tossWinnerSelect.appendChild(option2);
    }
}

function updateMatchDetails() {
    const team1 = document.getElementById('team1').value;
    const team2 = document.getElementById('team2').value;

    if (team1 && team2) {
        fetch(`/match_score?team1=${team1}&team2=${team2}`)
            .then(response => response.json())
            .then(data => {
                const detailsBox = document.getElementById('dataBox');
                detailsBox.style.display = 'block';
                document.getElementById('toss-winner-display').textContent = data.toss_winner;
                document.getElementById('toss-decision-display').textContent = data.toss_decision;
                document.getElementById('result-display').textContent = data.result;
                document.getElementById('margin-display').textContent = data.result_margin;
            })
            .catch(error => console.error('Error fetching match details:', error));
    }
}

function predict() {
    const team1 = document.getElementById('team1').value;
    const team2 = document.getElementById('team2').value;
    const city = document.getElementById('city').value;
    const requiredRuns = document.getElementById('required-runs').value;
    const requiredOvers = document.getElementById('required-overs').value;
    const requiredWickets = document.getElementById('required-wickets').value;

    if (!team1 || !team2 || team1 === team2) {
        showAlert('Please select two different teams.');
        return;
    }

    if (!city) {
        showAlert('Please select a city.');
        return;
    }

    if (!requiredRuns || requiredRuns <= 0) {
        showAlert('Please enter a valid number for required runs.');
        return;
    }

    if (!requiredOvers || requiredOvers <= 0) {
        showAlert('Please enter a valid number for required overs.');
        return;
    }

    if (!requiredWickets || requiredWickets < 0) {
        showAlert('Please enter a valid number for required wickets.');
        return;
    }

    const predictionData = {
        team1: team1,
        team2: team2,
        city: city,
        required_runs: requiredRuns,
        required_overs: requiredOvers,
        required_wickets: requiredWickets
    };

    fetch('/predict', {
        method: 'POST',
        headers: {
            'Content-Type': 'application/json'
        },
        body: JSON.stringify(predictionData)
    })
    .then(response => response.json())
    .then(data => {
        const result = data.team1_win_probability > 50
            ? `${team1} has a higher chance of winning!`
            : `${team2} has a higher chance of winning!`;

        const probability = data.team1_win_probability;
        displayPrediction(result, probability, team1, team2);

        scrollToProbabilitySection();
    })
    .catch(error => {
        console.error('Error during prediction:', error);
        showAlert('Error making prediction. Please try again later.');
    });
}

function displayPrediction(result, probability, team1Name, team2Name) {
    const winProbabilitySection = document.getElementById('win-probability');
    const probabilityChart = document.getElementById('probability-chart');
    probabilityChart.innerHTML = '';

    winProbabilitySection.style.display = 'block';

    const chartHTML = ` 
        <div class="probability-bar-container">
            <div class="team-name left">${team1Name}</div>
            <div class="probability-bar-background">
                <div class="probability-bar" style="width: ${probability}%;"></div>
            </div>
            <div class="team-name right">${team2Name}</div>
        </div>
        <div class="probability-text">
            <span>${probability}%</span> vs <span>${100 - probability}%</span>
        </div>`;

    probabilityChart.innerHTML = chartHTML;

    const resultText = document.createElement("p");
    resultText.textContent = result;
    resultText.style.textAlign = "center";
    resultText.style.fontWeight = "bold";
    resultText.style.color = "#333";
    winProbabilitySection.appendChild(resultText);
}

function scrollToProbabilitySection() {
    const winProbabilitySection = document.getElementById('win-probability');
    winProbabilitySection.scrollIntoView({
        behavior: 'smooth',
        block: 'start'
    });
}

window.onload = function() {
    updateTossOptions();
};
