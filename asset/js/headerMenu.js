document.addEventListener('DOMContentLoaded', function () {
    var normal = document.getElementById('navbarSupportedContent');
    if (normal) {
        var content = `
                    <ul class="navbar-nav me-auto mb-2 mb-lg-0" style="margin-left: auto; margin-right: 0;">
                        <li class="nav-item">
                            <a class="nav-link" href="/application/">Application</a>
                        </li>
                        <li class="nav-item">
                            <a class="nav-link" href="/application/demo/">Demo</a>
                        </li>
                        <li class="nav-item">
                            <a class="nav-link" href="/">Paper</a>
                        </li>
                        <li class="nav-item">
                            <a class="nav-link" href="/">Documentation</a>
                        </li>
                        <li class="nav-item">
                            <a class="nav-link" href="https://github.com/lygitdata/GarmentIQ" target="_blank">GitHub</a>
                        </li>
                        <li class="nav-item">
                            <a class="nav-link" href="/dashboard/">Dashboard</a>
                        </li>
                    </ul>
        `;
        normal.innerHTML = content;
    } else {
        console.error('Container not found. Make sure the div with id "normalMenu" exists.');
    }
});
