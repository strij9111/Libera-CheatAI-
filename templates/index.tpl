

<html>
    <link rel="stylesheet" href=https://cdnjs.cloudflare.com/ajax/libs/bulma/0.7.4/css/bulma.css />
	<style>
.spinner {
  opacity: 0.5;
  color: white;
  display: none;
  align-items: center;
  justify-content: center;
  min-height: 100vh;
            background-color:black;
            position:absolute;
            top:0px;
            left:0px;
            width:100%;
            height:100%;
            overflow:auto;
       
}

.spinner:after {
  animation: changeContent .8s linear infinite;
  display: block;
  content: "Думаем ⠋";
  font-size: 80px;
}

@keyframes changeContent {
  10% { content: "Думаем ⠙"; }
  20% { content: "Думаем ⠹"; }
  30% { content: "Думаем ⠸"; }
  40% { content: "Думаем ⠼"; }
  50% { content: "Думаем ⠴"; }
  60% { content: "Думаем ⠦"; }
  70% { content: "Думаем ⠧"; }
  80% { content: "Думаем ⠇"; }
  90% { content: "Думаем ⠏"; }
}
body {
    background-image:url('http://77.232.23.74/bg.png');
    background-size:100%;
}
	</style>

    <script defer src=https://use.fontawesome.com/releases/v5.3.1/js/all.js></script>	
    <head><title> Interactive Run </title></head>
    <body>
		<div class="spinner" id="loader"></div>
        <div class="columns" style="height: 100%">
            <div class="column is-three-fifths is-offset-one-fifth">
              <section class="hero is-info is-small has-text-grey-dark" style="background-color: transparent; height: 100%">
                <div id="parent" class="hero-body" style="overflow: auto; height: calc(100% - 76px); padding-top: 1em; padding-bottom: 0;">
                    <article class="media">
                      <div class="media-content">
                        <div class="content">
                          <p>
                            <strong>Инструкция</strong>
                            <br>
                            Введите ID пользователя (0 - новый пользователь)
                          </p>
                        </div>
                      </div>
                    </article>
                </div>
                <div class="hero-foot column is-three-fifths is-offset-one-fifth" style="height: 76px">
                  <form id = "interact">
                      <div class="field is-grouped">
                        <p class="control is-expanded">
                          <input class="input" type="text" id="userIn" placeholder="Введите ID пользователя...">
                        </p>
                        <p class="control">
                          <button id="respond" type="submit" class="button has-text-white-ter has-background-grey-dark">
                            Отправить
                          </button>
                        </p>
                       
                      </div>
                  </form>
                </div>
              </section>
            </div>
        </div>
        <script>
            document.getElementById("interact").addEventListener("submit", function(event){
                event.preventDefault()
                var text = document.getElementById("userIn").value;
                document.getElementById('userIn').value = "";

//				document.getElementById("loader").style.display = 'flex';
				
                fetch('/api?id=' + text, {
                    method: 'GET',
                }).then(response=> response.text()).then(data=>{
                    var parDiv = document.getElementById("parent");

                    // Change info for Model response
                    parDiv.innerHTML = data;
//					document.getElementById("loader").style.display = 'none';
					
//                    parDiv.scrollTo(0, parDiv.scrollHeight);
                })
            });
        </script>

    </body>
</html>
