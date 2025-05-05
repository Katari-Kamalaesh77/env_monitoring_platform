import React, { useEffect, useState } from 'react';
import { LineChart, Line, XAxis, YAxis, CartesianGrid, Tooltip } from 'recharts';

const Dashboard = () => {
  const [pm25, setPm25] = useState(null);
  const [pm25Category, setPm25Category] = useState('');
  const [o3, setO3] = useState(null);
  const [o3Category, setO3Category] = useState('');
  const [temperature, setTemperature] = useState(null);
  const [humidity, setHumidity] = useState(null);
  const [forecast, setForecast] = useState([]);

  // Fetching live air quality data
  useEffect(() => {
    fetch('http://127.0.0.1:8000/api/airquality')
      .then(response => response.json())
      .then(data => {
        setPm25(data.pm25);
        setPm25Category(data.pm25_category);
        setO3(data.o3);
        setO3Category(data.o3_category);
        setTemperature(data.temperature);
        setHumidity(data.humidity);
      })
      .catch(error => {
        console.error("Error fetching data:", error);
      });
  }, []);

  // Fetching PM2.5 forecast data
  useEffect(() => {
    fetch('http://127.0.0.1:8000/api/forecast/pm25')
      .then(res => res.json())
      .then(data => {
        // Check if the forecast data exists and has a correct structure
        if (data.forecast && Array.isArray(data.forecast)) {
          setForecast(data.forecast);
        }
      })
      .catch(error => {
        console.error("Error fetching forecast data:", error);
      });
  }, []);

  return (
    <div className="p-4">
      <h2 className="text-2xl font-bold mb-4">Air Quality Data</h2>
      <p><strong>PM2.5:</strong> {pm25 ?? 'Loading...'} ({pm25Category})</p>
      <p><strong>Ozone (O₃):</strong> {o3 ?? 'Loading...'} ({o3Category})</p>
      <p><strong>Temperature:</strong> {temperature ?? 'Loading...'}°F</p>
      <p><strong>Humidity:</strong> {humidity ?? 'Loading...'}%</p>

      {/* PM2.5 Forecast */}
      <h2 className="text-2xl font-bold mt-8 mb-4">PM2.5 Forecast</h2>
      {forecast.length > 0 ? (
        <LineChart width={600} height={300} data={forecast}>
          <CartesianGrid strokeDasharray="3 3" />
          <XAxis dataKey="ds" />
          <YAxis />
          <Tooltip />
          <Line type="monotone" dataKey="yhat" stroke="#8884d8" />
        </LineChart>
      ) : (
        <p>Loading forecast...</p>
      )}
    </div>
  );
};

export default Dashboard;
