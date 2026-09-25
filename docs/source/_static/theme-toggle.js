// Two-state light/dark toggle for the button in _templates/theme-toggle.html.
//
// The theme's own switcher offers light, dark and "auto" (follow the operating
// system); this site offers light and dark only, light by default. The mode is
// written to the attributes and localStorage keys the theme itself reads when a
// page loads, so the choice survives navigation and reloads.
(function () {
  function apply(mode) {
    document.documentElement.dataset.mode = mode;
    document.documentElement.dataset.theme = mode;
    localStorage.setItem("mode", mode);
    localStorage.setItem("theme", mode);
  }

  // A visitor who picked "auto" under the old switcher starts again in light.
  if (localStorage.getItem("mode") === "auto") {
    apply("light");
  }

  document.addEventListener("DOMContentLoaded", function () {
    document.querySelectorAll(".theme-toggle-button").forEach(function (button) {
      button.addEventListener("click", function () {
        apply(document.documentElement.dataset.theme === "dark" ? "light" : "dark");
      });
    });
  });
})();
