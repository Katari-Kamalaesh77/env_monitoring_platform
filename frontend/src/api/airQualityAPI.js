export async function getAirQualityData() {
  try {
    const response = await fetch('/api/airquality');  // Ensure this URL matches your backend URL
    if (!response.ok) {
      throw new Error('Failed to fetch air quality data');
    }
    return await response.json();
  } catch (error) {
    throw new Error(error.message);
  }
}
