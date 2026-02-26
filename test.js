import http from 'k6/http';
import { sleep } from 'k6';

export const options = {
  vus: 50,         // number of virtual users
  duration: '30s', // test duration
};

export default function () {
  http.post(
    'http://127.0.0.1:8000/predict',
    JSON.stringify({ text: "Let the dragon rides again on the winds of time" }),
    { headers: { 'Content-Type': 'application/json' } }
  );

  sleep(0.1);
}