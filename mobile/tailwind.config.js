/** @type {import('tailwindcss').Config} */
module.exports = {
  content: ["./app/**/*.{js,jsx,ts,tsx}", "./components/**/*.{js,jsx,ts,tsx}"],
  theme: {
    extend: {
      colors: {
        primary: "#1a7f37", // Matching the green from CareAtlas API landing
        primaryDark: "#135d28",
        danger: "#dc3545",
      },
    },
  },
  plugins: [],
};
