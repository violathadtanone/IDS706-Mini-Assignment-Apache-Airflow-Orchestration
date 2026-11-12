// Run with Playwright and Chromium available; exercises the real PostgreSQL run.
const { chromium } = require(process.env.PLAYWRIGHT_MODULE || 'playwright')
const assert = require('node:assert/strict')
const fs = require('node:fs')

async function main() {
  const output = process.env.UI_ARTIFACTS || '/workspace/output/ml/ui-validation'
  fs.mkdirSync(output, {recursive: true})
  const browser = await chromium.launch({executablePath: process.env.CHROMIUM_PATH,
    headless: true, args: ['--no-sandbox', '--disable-dev-shm-usage']})
  try {
    const page = await browser.newPage({baseURL: process.env.UI_URL || 'http://healthcare-ml:8001', viewport: {width: 1440, height: 1100}})
    const errors = []
    page.on('pageerror', error => errors.push(error.message))
    await page.goto(process.env.UI_URL || 'http://healthcare-ml:8001')
    await page.getByRole('button', {name: 'Start workflow run'}).waitFor()
    await page.waitForFunction(() => !document.querySelector('.run-control button').disabled)
    const started = page.waitForResponse(r => r.url().endsWith('/api/runs') && r.request().method() === 'POST')
    await page.getByRole('button', {name: 'Start workflow run'}).click()
    const {id} = await (await started).json()
    await page.locator('.agent.running').first().waitFor({timeout: 90000})
    await page.screenshot({path: `${output}/running.png`, fullPage: true})
    await page.locator('.badge.completed').waitFor({timeout: 180000})
    const run = await (await page.request.get(`/api/runs/${id}`)).json()
    assert.equal(run.status, 'completed')
    assert.equal(run.attempts.length, 3)
    assert.equal(await page.locator('tbody tr').count(), 3)
    assert.equal(run.result.selected_candidate, 'median_baseline')
    await page.getByRole('heading', {name: 'Median baseline selected'}).waitFor()
    for (const attempt of run.attempts) {
      assert.ok((await page.locator('tbody').innerText()).includes(attempt.validation_mae.toFixed(6)))
    }
    const finalMetrics = await page.locator('.result-metrics').innerText()
    assert.ok(finalMetrics.includes(run.result.test_mae.toFixed(6)))
    assert.ok(finalMetrics.includes(run.result.baseline_test_mae.toFixed(6)))
    assert.ok((await page.locator('.metrics').innerText()).includes(run.result.baseline_validation_mae.toFixed(6)))
    const path = run.events.filter(e => e.status === 'running').map(e => e.node)
    assert.deepEqual(path, run.result.path)
    assert.equal(path.filter(n => n === 'regularization_adjuster').length, 2)
    assert.equal(await page.locator('.agent.completed').count(), 6)
    const back = page.locator('[data-testid="rf__edge-regularization_adjuster->score_model_trainer"] path.react-flow__edge-path')
    assert.ok((await back.getAttribute('style')).includes('rgb(8, 126, 139)'))
    const text = await page.locator('body').innerText()
    assert.ok(!/\b(F1|accuracy|LLM|hypothesis|CSV upload)\b/i.test(text))
    assert.deepEqual(errors, [])
    const saved = JSON.parse(fs.readFileSync(`/workspace/output/ml/${id}/metrics.json`))
    assert.deepEqual(saved, run.result)
    await page.screenshot({path: `${output}/completed.png`, fullPage: true})
    await page.setViewportSize({width: 390, height: 844})
    await page.waitForTimeout(300) // allow the resize-only fitView to settle
    await page.screenshot({path: `${output}/mobile.png`, fullPage: true})
    fs.writeFileSync(`${output}/browser-evidence.json`, JSON.stringify({run_id: id, path,
      actual_live_status_observed: true, back_edge_traversals: 2, console_errors: errors,
      result: run.result}, null, 2))
    console.log(JSON.stringify({run_id: id, browser_validation: 'passed', result: run.result}, null, 2))
  } finally { await browser.close() }
}
main().catch(error => { console.error(error); process.exitCode = 1 })
