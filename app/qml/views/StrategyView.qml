import QtQuick
import QtQuick.Controls
import QtQuick.Layouts
import "../components"

Item {
    Layout.fillWidth: true
    Layout.fillHeight: true

    ColumnLayout {
        anchors.fill: parent
        anchors.margins: 10
        spacing: 10

        RowLayout {
            Layout.fillWidth: true
            TextField { id: search; placeholderText: "Search strategy name / id"; Layout.fillWidth: true }
            ComboBox { id: familyFilter; model: ["all", "trend", "breakout", "reversal", "flow_confirmed"] }
        }
        RowLayout {
            Layout.fillWidth: true
            spacing: 8
            Label { text: "Min Win %"; color: "#9FB5D7" }
            SpinBox { id: minWin; from: 0; to: 100; value: 0; stepSize: 1 }
            Label { text: "Max DD %"; color: "#9FB5D7" }
            SpinBox { id: maxDd; from: 1; to: 100; value: 100; stepSize: 1 }
            Label { text: "Min Trades"; color: "#9FB5D7" }
            SpinBox { id: minTrades; from: 0; to: 5000; value: 0; stepSize: 5 }
            CheckBox { id: validatedOnly; text: "Validated only"; checked: false }
            Item { Layout.fillWidth: true }
        }

        ScrollView {
            Layout.fillWidth: true
            Layout.fillHeight: true

            Column {
                width: parent.width
                spacing: 8
                Repeater {
                    model: appState.strategies
                    delegate: StrategyItem {
                        required property var modelData
                        width: parent.width
                        row: modelData
                        visible: {
                            var q = search.text.toLowerCase()
                            var passSearch = q.length === 0 || modelData.name.toLowerCase().indexOf(q) >= 0 || modelData.id.toLowerCase().indexOf(q) >= 0
                            var passFamily = familyFilter.currentText === "all" || modelData.family === familyFilter.currentText
                            var wr = Number(modelData.win_rate || 0)
                            var dd = Math.abs(Number(modelData.drawdown || 0))
                            var trades = Number(modelData.trade_count || 0)
                            var passWin = wr >= minWin.value
                            var passDd = dd <= maxDd.value
                            var passTrades = trades >= minTrades.value
                            var passValidated = !validatedOnly.checked || String(modelData.status) === "validated"
                            return passSearch && passFamily && passWin && passDd && passTrades && passValidated
                        }
                        selected: appState.selectedStrategy && appState.selectedStrategy.id === modelData.id
                        onClicked: function() { appState.selectStrategyById(modelData.id) }
                    }
                }
            }
        }
    }
}
